-----

# Automated News Broadcast Generator

This project creates an automated, continuous news broadcast stream by fetching articles from RSS feeds, processing them with advanced NLP techniques (summarization, sentiment analysis, relevancy scoring, and clustering), and then generating a dynamic news script. The script is then converted to speech and played, offering a personalized audio news experience.

-----

## Features

  * **Continuous News Stream:** Automatically fetches and processes news at a defined interval, providing an always up-to-date broadcast.
  * **Intelligent Article Curation:**
      * **Feed Fetching:** Asynchronously gathers articles from multiple RSS feeds.
      * **Content Extraction & Deduplication:** Cleans HTML content and skips duplicate articles using content hashing.
      * **Summarization:** Uses a local LLM (Ollama) to generate concise summaries for each article.
      * **Relevancy Scoring:** (Optional) Scores articles based on their relevance to a user-defined topic.
      * **Sentiment Analysis:** Analyzes the emotional tone of articles.
      * **Importance Scoring:** Ranks articles based on freshness, content quality, readability, and sentiment impact.
      * **TF-IDF Clustering:** Groups similar articles into coherent news topics.
  * **Dynamic Script Generation:**
      * Crafts news segments using an LLM, integrating summaries from clustered articles.
      * Allows for optional user guidance to influence script tone or focus.
  * **Real-time Audio Playback:**
      * Converts generated news scripts into natural-sounding speech using `edge-tts`.
      * Streams audio seamlessly in a separate thread for continuous playback.
  * **Robust Error Handling:** Implements a Circuit Breaker pattern for external API calls and comprehensive logging.
  * **Performance Monitoring:** Tracks key metrics like articles processed, processing times, and API call success rates.
  * **Persistent Caching:** Uses SQLite to cache processed articles, preventing redundant processing.

-----

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.8+**
  * **Ollama:** This project relies on a locally running Ollama instance to provide the language models for summarization, relevancy scoring, and script generation.
      * Download and install Ollama from [ollama.com](https://ollama.com/).
      * Pull the required models. The default configuration uses `mistral-small:24b-instruct-2501-q8_0` for summarization and broadcast generation, and `nomic-embed-text` for embeddings (though TF-IDF is currently used for clustering).
        ```bash
        ollama run mistral-small:24b-instruct-2501-q8_0
        ollama run nomic-embed-text
        ```
  * **NLTK Data:** The project uses NLTK for sentiment analysis. It will attempt to download the `vader_lexicon` automatically, but you can do it manually if needed:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kliewerdaniel/news08.git
    cd news08
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file is not provided in the prompt, but it should contain `aiohttp`, `feedparser`, `PyYAML`, `requests`, `pandas`, `numpy`, `scikit-learn`, `nltk`, `pydub`, and optionally `edge-tts`). If `edge-tts` is not available, audio playback will be disabled.

### Configuration

1.  **`feeds.yaml`:** Create a `feeds.yaml` file in the root directory of the project. This file will list the RSS feed URLs you want to monitor.

    Example `feeds.yaml`:

    ```yaml
    feeds:
      - https://www.reuters.com/arc/feed/rss/
      - https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml
      - https://feeds.bbci.co.uk/news/rss.xml
    ```

2.  **`CONFIG` in `news_generator.py`:** The `CONFIG` dictionary at the top of the `news_generator.py` file allows you to adjust various parameters:

      * `ollama_api`: Base URL for your Ollama instance.
      * `models`: Specify which Ollama models to use for different tasks.
      * `processing`: Control parameters like `max_articles_per_feed`, `min_article_length`, `max_clusters`, and `target_segments`.
      * `output`: Set `max_broadcast_length`.

-----

## Usage

Run the main script from your terminal:

```bash
python main.py
```

### Command-line Arguments

You can customize the broadcast behavior using command-line arguments:

  * `--topic <YOUR_TOPIC>`: (Optional) Focus the news generation on a specific topic (e.g., "artificial intelligence", "climate change"). Articles will be filtered based on their relevancy to this topic.
  * `--guidance <YOUR_GUIDANCE>`: (Optional) Provide guidance for refining the news script (e.g., "keep it optimistic", "focus on economic impacts").
  * `--fetch_interval <MINUTES>`: (Optional) Set the interval in minutes between fetching new news feeds (default: 15 minutes).

**Examples:**

  * To run a general news broadcast:
    ```bash
    python main.py
    ```
  * To get news focused on "Space Exploration":
    ```bash
    python main.py --topic "Space Exploration"
    ```
  * To get news with a "concise and formal" tone:
    ```bash
    python main.py --guidance "Keep the tone concise and formal."
    ```
  * To fetch news every 30 minutes:
    ```bash
    python main.py --fetch_interval 30
    ```

-----

## How it Works (Under the Hood)

1.  **Initialization:** The `NewsGenerator` class sets up logging, a SQLite database for caching, and NLTK for sentiment analysis. It also initializes a `CircuitBreaker` for API resilience and a `PerformanceMonitor`.
2.  **Feed Fetching (`fetch_feeds_batch`):**
      * Reads RSS feed URLs from `feeds.yaml`.
      * Asynchronously fetches content from feeds in batches using `aiohttp`.
      * Parses feed entries using `feedparser`, extracts content, and performs basic cleaning.
      * Checks for duplicate articles using a content hash against a 3-day cache in `news_cache.db`.
3.  **Article Processing (`process_articles_smart`):**
      * **Summarization:** For each new article, it calls the configured Ollama `summary_model` to generate a brief summary.
      * **Relevancy Scoring (if `--topic` is used):** Queries the LLM to score how relevant an article's title and summary are to the specified topic.
      * **Clustering (`cluster_articles_tfidf`):** Uses TF-IDF vectorization and K-Means clustering to group similar articles. This helps in creating coherent news segments.
      * **Importance Scoring (`calculate_importance_scores`):** Calculates a combined score for each article based on its freshness, content length, readability, and sentiment.
4.  **Broadcast Segment Creation (`create_broadcast_segments`):**
      * Groups articles by their assigned clusters.
      * If a topic is specified, filters articles below a defined relevancy threshold.
      * Selects the top articles from each cluster based on importance to form segments.
      * Determines a topic for each segment.
5.  **Script Generation (`generate_segment_script`):**
      * Uses the Ollama `broadcast_model` to generate a news anchor-style script for each segment, incorporating the summaries of the selected articles.
      * If `guidance` is provided, the script can be refined.
6.  **Audio Generation and Playback (`generate_and_queue_audio`, `play_audio_from_queue`):**
      * The `generate_and_queue_audio` function uses the `edge-tts` library to convert the generated script into an audio stream.
      * This audio data is then put into a `queue.Queue`.
      * A separate `player_thread` continuously pulls audio data from this queue and plays it using `pydub`. This ensures smooth, continuous playback without blocking the main news generation process.
7.  **Continuous Loop (`run_continuous`):** The `run_continuous` method orchestrates the entire process, running indefinitely at the specified `fetch_interval`.

-----

## Project Structure

```
.
├── main.py         # Main script containing the NewsGenerator class and logic
├── feeds.yaml                # Configuration file for RSS feed URLs (user-defined)
├── news.log                  # Log file for application events
├── news_cache.db             # SQLite database for caching processed article hashes
└── broadcast_logs/           # Directory to store Markdown logs of generated broadcasts
    └── log_YYYYMMDD_HHMMSS.md
```

-----

## Dependencies

The core dependencies for this project include:

  * `aiohttp`: For asynchronous HTTP requests (fetching feeds, communicating with Ollama).
  * `feedparser`: For parsing RSS/Atom feeds.
  * `PyYAML`: For loading configuration from `feeds.yaml`.
  * `requests`: (Potentially for legacy API calls, though `aiohttp` is preferred for async).
  * `pandas`, `numpy`: For data manipulation and numerical operations.
  * `scikit-learn`: For TF-IDF vectorization and K-Means clustering (`TfidfVectorizer`, `KMeans`).
  * `nltk`: For Natural Language Toolkit functionalities, specifically sentiment analysis (`SentimentIntensityAnalyzer`).
  * `pydub`: For audio manipulation and playback.
  * `edge-tts` (Optional): For text-to-speech generation. If not installed, audio features will be disabled.
  * `sqlite3`: Built-in Python library for database interactions.

-----

## Troubleshooting

  * **Ollama Connection Issues:** Ensure your Ollama server is running and accessible at the `base_url` specified in `CONFIG`. Check the Ollama logs for errors.
  * **Model Not Found:** Verify that you have pulled the required Ollama models (`mistral-small:24b-instruct-2501-q8_0`, `nomic-embed-text`) using `ollama run <model_name>`.
  * **Audio Playback:**
      * Confirm `edge-tts` is installed (`pip install edge-tts`).
      * Ensure your system has the necessary audio playback libraries that `pydub` relies on (e.g., `ffmpeg`). You might need to install `ffmpeg` separately if it's not already on your system.
      * Check the `news.log` file for any audio-related errors.
  * **Feed Fetching Errors:** Review `news.log` for specific HTTP errors or parsing issues related to the RSS feeds.
  * **High CPU/Memory Usage:** LLM inference can be resource-intensive. Consider using smaller models or adjusting `max_tokens` in the `CONFIG` for Ollama calls if you experience performance issues.

-----

## Contributing

Contributions are welcome\! If you have suggestions for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.







python -m spacy download en_core_web_sm
