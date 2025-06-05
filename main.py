import asyncio
import aiohttp
import feedparser
import yaml
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

# Check for optional dependencies
edge_tts_available = importlib.util.find_spec("edge_tts") is not None
if edge_tts_available:
    import edge_tts

# Streamlined configuration
CONFIG = {
    "ollama_api": {"base_url": "http://localhost:11434"},
    "models": {
        "summary_model": "mistral:latest",
        "broadcast_model": "mistral:latest",  # Simplified to use same model
        "embedding_model": "nomic-embed-text"
    },
    "processing": {
        "max_articles_per_feed": 8,
        "min_article_length": 100,
        "max_clusters": 10,
        "target_segments": 12
    },
    "output": {"max_broadcast_length": 10000}
}

@dataclass
class Article:
    title: str
    content: str
    url: str
    published: datetime
    source: str
    summary: str = ""
    sentiment_score: float = 0.0
    importance_score: float = 0.0
    relevancy_score: float = 0.0 # New field for relevancy
    cluster_id: int = -1

@dataclass
class BroadcastSegment:
    topic: str
    content: str
    articles: List[Article]
    importance: float

class CircuitBreaker:
    """Simple circuit breaker for external services"""
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "OPEN":
            if datetime.now().timestamp() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class PerformanceMonitor:
    """Track system performance metrics"""
    def __init__(self):
        self.metrics = {
            'articles_processed': 0,
            'processing_times': [],
            'api_calls': 0,
            'errors': 0
        }

    def track_operation(self, operation: str, duration: float, success: bool = True):
        self.metrics['processing_times'].append(duration)
        if not success:
            self.metrics['errors'] += 1

    def get_stats(self) -> Dict:
        if not self.metrics['processing_times']:
            return self.metrics

        times = self.metrics['processing_times']
        return {
            **self.metrics,
            'avg_processing_time': np.mean(times),
            'total_time': sum(times),
            'success_rate': 1 - (self.metrics['errors'] / len(times)) if times else 1
        }

class NewsGenerator:
    def __init__(self, feeds_file: str = "feeds.yaml", topic: Optional[str] = None, guidance: Optional[str] = None):
        self.feeds_file = feeds_file
        self.db_path = "news_cache.db"
        self.circuit_breaker = CircuitBreaker()
        self.performance_monitor = PerformanceMonitor()
        self.topic = topic
        self.guidance = guidance
        self.relevancy_threshold = 5 # Default threshold
        self.setup_logging()
        self.setup_database()
        self.setup_nlp()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('news.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                content_hash TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def setup_nlp(self):
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            self.logger.warning("NLTK setup failed")
            self.sentiment_analyzer = None

    async def fetch_feeds_batch(self, batch_size: int = 5) -> List[Article]:
        """Optimized batch processing of feeds"""
        with open(self.feeds_file, 'r') as f:
            feeds_config = yaml.safe_load(f)

        feeds = feeds_config.get('feeds', [])
        articles = []

        # Process feeds in batches
        for i in range(0, len(feeds), batch_size):
            batch = feeds[i:i+batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_single_feed(session, feed) for feed in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, list):
                        articles.extend(result)

            # Small delay between batches to be respectful
            if i + batch_size < len(feeds):
                await asyncio.sleep(0.5)

        return articles

    async def fetch_single_feed(self, session: aiohttp.ClientSession, feed_url: str) -> List[Article]:
        """Streamlined feed fetching"""
        self.logger.info(f"Attempting to fetch feed: {feed_url}")
        try:
            async with session.get(feed_url, timeout=60) as response:
                response.raise_for_status() # Raise an exception for HTTP errors
                content = await response.text()

            feed = feedparser.parse(content)
            articles = []

            fetched_count = 0
            for entry in feed.entries:
                if fetched_count >= CONFIG["processing"]["max_articles_per_feed"]:
                    break

                content = self.extract_content(entry)
                if len(content) < CONFIG["processing"]["min_article_length"]:
                    self.logger.debug(f"Skipping article due to short length: {entry.get('title', 'No Title')}")
                    continue

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if self.is_duplicate(content_hash):
                    self.logger.debug(f"Skipping duplicate article: {entry.get('title', 'No Title')}")
                    continue

                article = Article(
                    title=entry.get('title', ''),
                    content=content,
                    url=entry.get('link', ''),
                    published=self.parse_date(entry),
                    source=feed.feed.get('title', feed_url),
                )

                articles.append(article)
                self.cache_article(content_hash)
                fetched_count += 1

            self.logger.info(f"Successfully fetched {len(articles)} articles from {feed_url}")
            return articles

        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error fetching {feed_url}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"General error fetching or parsing {feed_url}: {e}")
            return []

    def extract_content(self, entry) -> str:
        """Extract and clean content"""
        content = ""
        for field in ['content', 'summary', 'description']:
            if hasattr(entry, field):
                if field == 'content' and entry.content:
                    content = entry.content[0].value if entry.content else ""
                else:
                    content = getattr(entry, field, "")
                break

        return re.sub(r'<[^>]+>', '', content).strip()

    def parse_date(self, entry) -> datetime:
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                return datetime(*entry.published_parsed[:6])
        except:
            pass
        return datetime.now()

    def is_duplicate(self, content_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT 1 FROM articles WHERE content_hash = ? AND processed_at > ?",
            (content_hash, datetime.now() - timedelta(days=3))
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def cache_article(self, content_hash: str):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO articles (id, content_hash) VALUES (?, ?)",
                (content_hash, content_hash)
            )
            conn.commit()
        except Exception as e:
            self.logger.error(f"Cache error: {e}")
        finally:
            conn.close()

    async def process_articles_smart(self, articles: List[Article]) -> List[Article]:
        """Streamlined processing with circuit breaker"""
        if not articles:
            return []

        start_time = datetime.now()

        async with aiohttp.ClientSession() as session:
            # Generate summaries
            summary_tasks = [self.generate_summary_safe(session, article) for article in articles]
            summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)

            for i, result in enumerate(summaries):
                if isinstance(result, Exception):
                    self.logger.error(f"Summary failed for {articles[i].title}: {result}")
                    articles[i].summary = articles[i].content[:150] + "..."
                else:
                    articles[i].summary = result

            # Calculate relevancy scores if a topic is provided
            if self.topic:
                relevancy_tasks = [self.calculate_relevancy_score(session, article) for article in articles]
                relevancy_scores = await asyncio.gather(*relevancy_tasks, return_exceptions=True)

                for i, result in enumerate(relevancy_scores):
                    if isinstance(result, Exception):
                        self.logger.error(f"Relevancy scoring failed for {articles[i].title}: {result}")
                        articles[i].relevancy_score = 0.0 # Default to 0 on failure
                    else:
                        articles[i].relevancy_score = result

        # Fast clustering using TF-IDF (more reliable than API embeddings)
        self.cluster_articles_tfidf(articles)

        # Calculate importance scores
        self.calculate_importance_scores(articles)

        duration = (datetime.now() - start_time).total_seconds()
        self.performance_monitor.track_operation("process_articles", duration)

        return articles

    async def generate_summary_safe(self, session: aiohttp.ClientSession, article: Article) -> str:
        """Safe summary generation with circuit breaker using aiohttp"""
        if not self.circuit_breaker.can_execute():
            return article.content[:150] + "..."

        prompt = f"Summarize in 2 sentences: {article.title}\n{article.content[:500]}"

        try:
            async with session.post(
                f"{CONFIG['ollama_api']['base_url']}/api/generate",
                json={
                    'model': CONFIG["models"]["summary_model"],
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.3, 'max_tokens': 100}
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                self.circuit_breaker.record_success()
                return data['response'].strip()

        except aiohttp.ClientResponseError as e:
            error_detail = await e.response.text() if e.response else "No response body"
            self.logger.error(f"Summary LLM API error (status: {e.status}): {error_detail}")
            self.circuit_breaker.record_failure()
            article.summary = article.content[:150] + "..."
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during summary LLM call: {e}")
            self.circuit_breaker.record_failure()
            article.summary = article.content[:150] + "..."
        except Exception as e:
            self.logger.error(f"Unexpected error during summary LLM call: {e}")
            self.circuit_breaker.record_failure()
            article.summary = article.content[:150] + "..."

    async def calculate_relevancy_score(self, session: aiohttp.ClientSession, article: Article) -> float:
        """Calculate relevancy score using LLM with aiohttp"""
        if not self.topic or not self.circuit_breaker.can_execute():
            return 0.0 # Default to 0 if no topic or circuit open

        prompt = f"""Given the topic: "{self.topic}"

        Score the following article from 0 to 10 on how relevant it is to the topic.
        Return only the score as a single number.

        Article Title: {article.title}
        Article Summary: {article.summary}
        """

        try:
            async with session.post(
                f"{CONFIG['ollama_api']['base_url']}/api/generate",
                json={
                    'model': CONFIG["models"]["summary_model"], # Using summary model for scoring
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1, 'max_tokens': 5} # Low temperature for score
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                self.circuit_breaker.record_success()
                score_str = data['response'].strip()
                try:
                    score = float(score_str)
                    return max(0.0, min(10.0, score)) # Ensure score is between 0 and 10
                except ValueError:
                    self.logger.warning(f"LLM returned non-numeric relevancy score: '{score_str}'")
                    return 0.0

        except aiohttp.ClientResponseError as e:
            error_detail = await e.response.text() if e.response else "No response body"
            self.logger.error(f"Relevancy scoring LLM API error (status: {e.status}): {error_detail}")
            self.circuit_breaker.record_failure()
            article.relevancy_score = 0.0
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during relevancy scoring LLM call: {e}")
            self.circuit_breaker.record_failure()
            article.relevancy_score = 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error during relevancy scoring LLM call: {e}")
            self.circuit_breaker.record_failure()
            article.relevancy_score = 0.0

    def cluster_articles_tfidf(self, articles: List[Article]):
        """Fast TF-IDF based clustering"""
        if len(articles) < 2:
            return

        texts = [f"{article.title} {article.summary}" for article in articles]

        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            embeddings = vectorizer.fit_transform(texts).toarray()

            n_clusters = min(CONFIG["processing"]["max_clusters"], len(articles))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings)

            for article, label in zip(articles, cluster_labels):
                article.cluster_id = int(label)

        except Exception as e:
            self.logger.error(f"Clustering error: {e}")
            for i, article in enumerate(articles):
                article.cluster_id = i % 3

    def calculate_importance_scores(self, articles: List[Article]):
        """Enhanced importance scoring"""
        for article in articles:
            # Freshness (0-1)
            hours_old = (datetime.now() - article.published).total_seconds() / 3600
            freshness = max(0, 1 - (hours_old / 48))

            # Content quality (0-1)
            content_quality = min(1.0, len(article.content) / 800)

            # Readability score (simple measure)
            sentences = len(re.split(r'[.!?]+', article.content))
            words = len(article.content.split())
            readability = 1.0 if sentences == 0 else min(1.0, words / (sentences * 15))

            # Sentiment impact
            if self.sentiment_analyzer:
                sentiment = self.sentiment_analyzer.polarity_scores(article.content)
                article.sentiment_score = sentiment['compound']
                sentiment_impact = abs(sentiment['compound'])
            else:
                sentiment_impact = 0.5

            # Combined score
            article.importance_score = (
                0.4 * freshness +
                0.3 * content_quality +
                0.2 * sentiment_impact +
                0.1 * readability
            )

    def create_broadcast_segments(self, articles: List[Article]) -> List[BroadcastSegment]:
        """Create segments from clustered articles"""
        segments = []
        clusters = {}

        # Filter articles by relevancy threshold if a topic is provided
        if self.topic:
            articles = [a for a in articles if a.relevancy_score >= self.relevancy_threshold]
            self.logger.info(f"Filtered to {len(articles)} articles above relevancy threshold ({self.relevancy_threshold}) for topic '{self.topic}'")
            if not articles:
                self.logger.warning("No articles met the relevancy threshold for the given topic.")
                return []

        for article in articles:
            cluster_id = article.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(article)

        for cluster_id, cluster_articles in clusters.items():
            if not cluster_articles:
                continue

            cluster_articles.sort(key=lambda x: x.importance_score, reverse=True)
            topic = self.extract_topic(articles[:2])
            selected_articles = cluster_articles[:2]
            avg_importance = np.mean([a.importance_score for a in selected_articles])

            segment = BroadcastSegment(
                topic=topic,
                content="",
                articles=selected_articles,
                importance=avg_importance
            )
            segments.append(segment)

        segments.sort(key=lambda x: x.importance, reverse=True)
        return segments[:CONFIG["processing"]["target_segments"]]

    def extract_topic(self, articles: List[Article]) -> str:
        """Simple topic extraction"""
        if not articles:
            return "General News"

        # If a specific topic is provided, use it
        if self.topic:
            return self.topic

        # Extract key words from titles
        all_words = []
        for article in articles:
            words = re.findall(r'\b[A-Z][a-z]+\b', article.title)
            all_words.extend(words)

        if all_words:
            # Return most common capitalized word
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            return max(word_counts, key=word_counts.get)

        return articles[0].title.split()[:2]

    async def generate_broadcast_script(self, segments: List[BroadcastSegment]) -> str:
        """Generate professional broadcast script"""
        script_parts = []
        current_time = datetime.now().strftime("%A, %B %d")

        # Opening
        script_parts.append(f"Today's news briefing for {current_time}.")

        for segment in segments:
            segment_script = await self.generate_segment_script(segment)
            segment.content = segment_script
            script_parts.append(segment_script)

        # Closing
        script_parts.append("That concludes today's news update.")

        full_script = " ".join(script_parts)

        # Trim if needed
        if len(full_script) > CONFIG["output"]["max_broadcast_length"]:
            full_script = full_script[:CONFIG["output"]["max_broadcast_length"]]

        async with aiohttp.ClientSession() as session:
            # Refinement prompt
            if self.guidance:
                self.logger.info("Applying refinement prompt to the full script.")
                full_script = await self.refine_script(session, full_script, self.guidance)

        return self.clean_script_for_tts(full_script)

    async def generate_segment_script(self, segment: BroadcastSegment) -> str:
        """Generate segment script"""
        context = "\n".join([f"{a.title}: {a.summary}" for a in segment.articles])

        prompt = f"""Write a news segment about {segment.topic}. Use this information:
{context}

Write 2-3 sentences in news anchor style. Start directly with the news:"""

        # Add guidance to the prompt if available
        if self.guidance:
            prompt += f"\n\nGuidance for script generation: {self.guidance}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONFIG['ollama_api']['base_url']}/api/generate",
                    json={
                        'model': CONFIG["models"]["broadcast_model"],
                        'prompt': prompt,
                        'stream': False,
                        'options': {'temperature': 0.4, 'max_tokens': 300}
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['response'].strip()

        except aiohttp.ClientResponseError as e:
            error_detail = await e.response.text() if e.response else "No response body"
            self.logger.error(f"Script generation LLM API error (status: {e.status}): {error_detail}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during script generation LLM call: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during script generation LLM call: {e}")

        # Fallback
        if segment.articles:
            return f"In {segment.topic} news, {segment.articles[0].title}."
        return f"Updates on {segment.topic}."

    async def refine_script(self, session: aiohttp.ClientSession, script: str, guidance: str) -> str:
        """Refine the generated script based on guidance using aiohttp"""
        prompt = f"""Refine the following news script based on the provided guidance.

        Original Script:
        {script}

        Guidance for refinement:
        {guidance}

        Return the refined script:"""

        try:
            async with session.post(
                f"{CONFIG['ollama_api']['base_url']}/api/generate",
                json={
                    'model': CONFIG["models"]["broadcast_model"], # Use broadcast model for refinement
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.5, 'max_tokens': CONFIG["output"]["max_broadcast_length"]}
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.info("Script refined successfully.")
                return data['response'].strip()

        except aiohttp.ClientResponseError as e:
            error_detail = await e.response.text() if e.response else "No response body"
            self.logger.error(f"Script refinement LLM API error (status: {e.status}): {error_detail}")
            return script # Return original script on error
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during script refinement LLM call: {e}")
            return script # Return original script on error
        except Exception as e:
            self.logger.error(f"Unexpected error during script refinement LLM call: {e}")
            return script # Return original script on error

    def clean_script_for_tts(self, script: str) -> str:
        """Clean script for TTS"""
        script = re.sub(r'[#*_~`|]', '', script)
        script = re.sub(r'https?://\S+', '', script)
        script = script.replace('%', ' percent ').replace('&', ' and ')
        return re.sub(r'\s+', ' ', script).strip()

    async def generate_audio_smart(self, script: str, output_file: str):
        """Audio generation using Edge TTS."""
        try:
            if not edge_tts_available:
                raise ModuleNotFoundError("Edge TTS is not available. Please install it using 'pip install edge-tts'.")

            voice = "en-US-JennyNeural"
            communicate = edge_tts.Communicate(script, voice)
            await communicate.save(output_file)
            self.logger.info(f"Audio generated with Edge TTS: {output_file}")

        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            raise

    def save_results(self, script: str, segments: List[BroadcastSegment], output_file: str):
        """Save enhanced results"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = self.performance_monitor.get_stats()

        content = f"""# News Broadcast - {timestamp}

## Performance Stats
- Articles: {stats['articles_processed']}
- Success Rate: {stats['success_rate']:.1%}
- Processing Time: {stats.get('total_time', 0):.1f}s

## Script
{script}

## Sources
"""

        for i, segment in enumerate(segments, 1):
            content += f"\n### {i}. {segment.topic} (Score: {segment.importance:.2f})\n"
            for article in segment.articles:
                content += f"- {article.title} ({article.source}) - Relevancy: {article.relevancy_score:.1f}/10\n"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

    async def run(self):
        """Streamlined main pipeline"""
        start_time = datetime.now()
        self.logger.info("Starting news generation pipeline")

        try:
            # Fetch and process
            articles = await self.fetch_feeds_batch()
            self.logger.info(f"Fetched {len(articles)} articles")
            self.performance_monitor.metrics['articles_processed'] = len(articles)

            if not articles:
                self.logger.warning("No articles found")
                return

            processed_articles = await self.process_articles_smart(articles)
            segments = self.create_broadcast_segments(processed_articles)
            script = await self.generate_broadcast_script(segments)

            # Generate outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            md_file = f"news_{timestamp}.md"
            mp3_file = f"news_{timestamp}.mp3"

            self.save_results(script, segments, md_file)
            await self.generate_audio_smart(script, mp3_file)

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pipeline completed in {duration:.1f}s")
            self.logger.info(f"Files: {md_file}, {mp3_file}")

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate news broadcasts from RSS feeds.")
    parser.add_argument("--topic", type=str, help="Optional topic to filter and focus news generation.")
    parser.add_argument("--guidance", type=str, help="Optional guidance for refining the news script.")
    parser.add_argument("--relevancy_threshold", type=int, default=5,
                        help="Minimum relevancy score (0-10) for articles to be included when a topic is provided. Default is 5.")

    args = parser.parse_args()

    generator = NewsGenerator(topic=args.topic, guidance=args.guidance)
    generator.relevancy_threshold = args.relevancy_threshold

    asyncio.run(generator.run())

if __name__ == "__main__":
    main()