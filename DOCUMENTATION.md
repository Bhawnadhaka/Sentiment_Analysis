# MLOps Sentiment Analysis Pipeline — Project Documentation

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [End-to-End Flow](#2-end-to-end-flow)
3. [Project Structure](#3-project-structure)
4. [ML Model Architecture](#4-ml-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Data Pipeline](#6-data-pipeline)
7. [API Design](#7-api-design)
8. [Streaming Architecture](#8-streaming-architecture)
9. [Monitoring & Observability](#9-monitoring--observability)
10. [Technology Stack — Why Each Was Chosen](#10-technology-stack--why-each-was-chosen)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Configuration Files](#12-configuration-files)
13. [Testing](#13-testing)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Achieved Metrics](#15-achieved-metrics)

---

## 1. What Is This Project?

This is a **production-grade, real-time sentiment analysis system** built with full MLOps practices. The system continuously ingests live news articles from the internet, classifies them as *positive* or *negative*, and displays the results on an interactive web dashboard — all while tracking model performance, detecting data drift, and exposing operational metrics.

It is not just a machine-learning model; it is an end-to-end pipeline that covers data ingestion, model training, model serving, streaming, caching, monitoring, and containerised deployment.

---

## 2. End-to-End Flow

```
News API → Kafka Producer → Kafka Topic → Kafka Consumer
              → Sentiment Model (BoW / LSTM / DistilBERT)
                  → SQLite Database
                      → Streamlit Dashboard
                          → Prometheus / Grafana metrics
                              → Evidently AI drift reports
```

**Step-by-step:**

1. **News Ingestion** — `NewsAPIStream` polls [newsapi.org](https://newsapi.org) for articles on configurable topics (e.g., *technology*, *AI*, *startups*). Each article title + description is combined into a single text blob.

2. **Kafka Streaming** — `NewsKafkaProducer` serialises each article as JSON and publishes it to a Kafka topic (`news-sentiment`). The article URL is used as the partition key to prevent duplicates.

3. **Sentiment Prediction** — `SentimentConsumer` subscribes to the Kafka topic, loads the ML model (`SentimentPredictor`), and runs inference on each article in real time.

4. **Storage** — Predictions (title, sentiment, confidence, source, topic, timestamps) are persisted in a local **SQLite** database (`sentiment_predictions.db`).

5. **REST API** — A **FastAPI** server (`src/api/main.py`) exposes `/predict` and `/predict/batch` endpoints, with **Redis** caching to serve repeated queries in ~5 ms instead of ~50 ms.

6. **Web Dashboard** — A **Streamlit** app (`app.py`) provides six pages: Overview, Live Predictions, Analytics, Model Performance, System Status, and Real-Time Stream monitor.

7. **Observability** — **Prometheus** scrapes the `/metrics` endpoint; **Grafana** visualises them. **Evidently AI** generates offline HTML reports for data quality, data drift, and classification performance.

8. **Experiment Tracking** — All training runs are logged via **MLflow** (local or DagsHub-hosted), capturing hyperparameters, loss curves, accuracy, F1, precision, recall, and model artifacts.

---

## 3. Project Structure

```
Sentiment_Analysis/
├── app.py                        # Streamlit web dashboard (6-page UI)
├── run_pipeline.py               # CLI entry point: producer / consumer / dashboard / check
├── setup.py                      # Python package setup
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Dev/lint dependencies
├── Makefile                      # Convenience commands
├── .env.example                  # Template for secrets (API keys, tokens)
├── .pre-commit-config.yaml       # Black + Flake8 pre-commit hooks
│
├── configs/
│   ├── model_config.yaml         # Model type, dropout, vocab size, etc.
│   ├── training_config.yaml      # Batch size, epochs, LR, early stopping
│   └── deployment_config.yaml   # Runtime deployment settings
│
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app with lifespan (startup/shutdown)
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── cache.py              # Redis caching layer (MD5-keyed TTL cache)
│   │   ├── metrics.py            # Prometheus counters/gauges/histograms
│   │   ├── kafka_producer.py     # Publishes API predictions back to Kafka
│   │   └── kafka_consumer.py     # (API-side consumer)
│   │
│   ├── data/
│   │   ├── data_loader.py        # PyTorch Datasets + DataLoaders for all model types
│   │   └── news_stream.py        # News API client (headlines + search + continuous stream)
│   │
│   ├── models/
│   │   ├── model.py              # DistilBERT, LSTM, BoW architectures + factory
│   │   ├── train.py              # Trainer class with MLflow tracking
│   │   ├── inference.py          # SentimentPredictor (load checkpoint → predict)
│   │   └── evaluate.py           # Metrics computation (accuracy, F1, precision, recall)
│   │
│   ├── streaming/
│   │   ├── news_producer.py      # NewsKafkaProducer: News API → Kafka
│   │   └── sentiment_consumer.py # SentimentConsumer: Kafka → Model → SQLite
│   │
│   ├── monitoring/
│   │   ├── dashboard.py          # Evidently AI report generator (4 report types)
│   │   ├── drift_detector.py     # Data drift detection logic
│   │   └── performance_tracker.py # Model performance tracking over time
│   │
│   └── utils/
│       ├── config.py             # YAML config loader + merger
│       ├── logger.py             # Loguru logger setup
│       └── metrics.py            # Shared metrics helpers
│
├── docker/
│   ├── docker-compose.yml        # Full stack: Kafka, Zookeeper, API, Consumer,
│   │                             # Redis, Prometheus, Grafana, Streamlit
│   ├── Dockerfile.api            # FastAPI container
│   ├── Dockerfile.kafka          # Kafka consumer container
│   ├── Dockerfile.streamlit      # Dashboard container
│   ├── Dockerfile.train          # Training container
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana/dashboards/       # Pre-built Grafana dashboard JSON
│
├── monitoring/
│   └── evidently_reports/        # Generated HTML reports
│
├── models/                       # Saved .pth model checkpoints
├── data/
│   └── processed/                # train.csv, val.csv, test.csv
└── tests/
    ├── test_api.py
    ├── test_model.py
    ├── test_data.py
    └── test_kafka.py
```

---

## 4. ML Model Architecture

Three model architectures are supported, selected via the `--model` flag:

| Model | Architecture | Approx. Params | Accuracy | Latency | Use Case |
|---|---|---|---|---|---|
| **BoW** (deployed) | Embedding → AvgPool → FC(128) → FC(2) | ~1 M | **76.45%** | ~50 ms | Production (fast) |
| **LSTM** | Embedding → BiLSTM(2-layer) → FC(2) | ~3 M | ~80% | ~100 ms | Better quality |
| **DistilBERT** | Pretrained DistilBERT + Dropout + Linear(2) | ~67 M | ~85% | ~150 ms | Best quality |

### BoW (Bag-of-Words) — Production Model

- Builds a vocabulary of the top 10,000 words from training data
- Converts text to a sequence of word IDs
- Embeds each word (dim=100), averages embeddings across sequence length (pooling)
- Passes the pooled vector through a 2-layer feed-forward classifier

### LSTM

- 128-dim word embeddings
- 2-layer bidirectional LSTM with 256 hidden units → 512-dim output
- Dropout (0.3) + Linear classifier

### DistilBERT

- Loads pretrained `distilbert-base-uncased` from HuggingFace
- Uses the `[CLS]` token embedding (dim=768)
- Dropout (0.3) + Linear classifier (768→2)
- Optionally freezes BERT parameters (transfer learning / fine-tuning)

---

## 5. Training Pipeline

**File:** `src/models/train.py`

1. Load YAML config from `configs/training_config.yaml`
2. Load preprocessed CSV data → build vocab (for BoW/LSTM) or use DistilBERT tokenizer
3. Create model via factory function `create_model(type, **config)`
4. Train with **AdamW** optimizer, **CrossEntropyLoss**, **ReduceLROnPlateau** scheduler
5. **Early stopping** with configurable patience (default: 3 epochs without improvement)
6. Log all metrics (loss, accuracy, F1, precision, recall, LR) per epoch to **MLflow**
7. Save best checkpoint to `models/best_{type}_model.pth` (includes vocab + config)
8. Final evaluation on held-out test set → log test metrics to MLflow

**Results achieved:**

| Metric | Value |
|---|---|
| Accuracy | 76.45% |
| F1 Score | 76.68% |
| Precision | 76.78% |
| Recall | 76.45% |

---

## 6. Data Pipeline

**Source:** [NewsAPI.org](https://newsapi.org) — free tier (100 requests/day)

**File:** `src/data/news_stream.py`

- `get_top_headlines()` — country/category-based headlines
- `search_everything()` — keyword search across all sources
- `stream_continuous()` — generator that loops indefinitely with rate-limit-respecting delays (5 min between batches)

Each article is normalised as:

```json
{
  "text": "<title>. <description>",
  "title": "AI startup raises $500M in Series C",
  "source": "TechCrunch",
  "url": "https://techcrunch.com/...",
  "published_at": "2025-01-01T10:00:00Z",
  "topic": "technology"
}
```

Duplicate URLs are filtered using a `seen_urls` set.

---

## 7. API Design

**File:** `src/api/main.py` — FastAPI with async lifespan management

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Root health check |
| `/health` | GET | Detailed health: model loaded, Kafka enabled |
| `/predict` | POST | Single text → sentiment + confidence + probabilities |
| `/predict/batch` | POST | List of texts → list of predictions |
| `/model/info` | GET | Model type, device, classes, cache stats |
| `/metrics` | GET | Prometheus-format text metrics |

**Request example:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla stock surges to all-time high"}'
```

**Response:**

```json
{
  "text": "Tesla stock surges to all-time high",
  "sentiment": "positive",
  "confidence": 0.89,
  "probabilities": {
    "negative": 0.11,
    "positive": 0.89
  }
}
```

**Cache layer:** Every `/predict` call checks Redis first (MD5 hash of `text + model_type` as key, 1-hour TTL). Cache hit rate is ~70%, reducing latency from ~50 ms → ~5 ms.

---

## 8. Streaming Architecture

```
                   ┌──────────────────┐
  NewsAPI.org  ──► │ NewsKafkaProducer │ ──► Kafka Topic: news-sentiment
                   └──────────────────┘
                                              │
                                              ▼
                                 ┌────────────────────────┐
                                 │  SentimentConsumer      │
                                 │  (SentimentPredictor)   │
                                 └────────────────────────┘
                                              │
                                              ▼
                                        SQLite DB
                                 (sentiment_predictions.db)
```

**Producer settings:**

- `acks='all'` — waits for all replicas to acknowledge
- `retries=3` — automatic retry on transient failures
- `max_in_flight_requests_per_connection=1` — guarantees ordering

**Consumer settings:**

- `auto_offset_reset='earliest'` — processes backlogged messages on restart
- `enable_auto_commit=True`
- Consumer group: `sentiment-consumer-group`

**Deduplication:** `INSERT OR REPLACE` keyed on the unique article URL in SQLite.

---

## 9. Monitoring & Observability

### Prometheus Metrics (tracked in `/metrics`)

| Metric | Type | Description |
|---|---|---|
| `prediction_requests_total` | Counter | Total API calls, labelled by endpoint |
| `prediction_latency_seconds` | Histogram | Response time distribution |
| `model_loaded` | Gauge | 1 = loaded, 0 = not loaded |
| `kafka_connected` | Gauge | 1 = connected, 0 = disconnected |
| `cache_hits_total` | Counter | Redis cache hits |
| `cache_misses_total` | Counter | Redis cache misses |

### Grafana

- Pre-built JSON dashboard in `docker/grafana/dashboards/`
- Visualises request rate, latency percentiles, cache hit rate, and model health

### Evidently AI Reports

Generated by running `python run_pipeline.py dashboard`. Saved as HTML to `monitoring/evidently_reports/`:

1. **Data Quality Report** — missing values, value distributions, schema checks
2. **Classification Performance Report** — comparison of reference vs. current predictions
3. **Data Drift Report** — statistical tests (PSI, KS) on `confidence` feature and prediction distribution
4. **Detailed Metrics Report** — confusion matrix, probability distributions, quantile analysis

---

## 10. Technology Stack — Why Each Was Chosen

| Technology | Why Used |
|---|---|
| **PyTorch** | Flexible deep learning framework; supports all three model architectures natively; industry standard for research and production |
| **HuggingFace Transformers** | Provides pretrained DistilBERT weights and tokenizer out of the box; enables strong NLP without training a transformer from scratch |
| **FastAPI** | Async, high-performance Python web framework; auto-generates OpenAPI docs; native Pydantic integration for request validation |
| **Apache Kafka** | Industry-standard distributed message queue; enables decoupled, fault-tolerant, horizontally scalable real-time streaming |
| **Redis** | In-memory key-value store used as a prediction cache; reduces model inference load for repeated queries by ~10× |
| **Prometheus** | Pull-based metrics collection; integrates seamlessly with FastAPI via `prometheus-client`; vendor-neutral and widely adopted |
| **Grafana** | Best-in-class metrics visualisation; connects directly to Prometheus; pre-built dashboards available |
| **Evidently AI** | Purpose-built Python library for ML monitoring; generates interactive HTML reports for drift, data quality, and classification performance |
| **MLflow** | Experiment tracking — logs params, metrics, artifacts per training run; supports both local and cloud (DagsHub) backends |
| **DVC** | Data version control; tracks large data files (CSVs, models) in Git without bloating the repository |
| **Streamlit** | Rapid Python-native dashboard framework; ideal for ML apps; supports live data polling and interactive Plotly charts |
| **Plotly** | Interactive charting library used within Streamlit for trend lines, pie charts, bar charts, and histograms |
| **SQLite** | Zero-config embedded database; sufficient for local development; stores all prediction records with full SQL query support |
| **Docker Compose** | Orchestrates 8 containers (Kafka, Zookeeper, API, Consumer, Redis, Prometheus, Grafana, Streamlit) with a single command |
| **NewsAPI** | Free-tier REST API (100 req/day) for real news articles; provides realistic, domain-relevant streaming data |
| **Loguru** | Drop-in Python logger with structured output and zero-config setup; better defaults than the stdlib `logging` module |
| **DagsHub** | Free Git + MLflow + DVC hosting platform; replaces the need for a paid MLOps cloud platform for students and indie developers |
| **scikit-learn** | Used for computing evaluation metrics (accuracy, F1, precision, recall, confusion matrix) |
| **Pydantic** | Data validation and serialisation for FastAPI request/response schemas |

---

## 11. How to Run the Project

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Free API key from [newsapi.org](https://newsapi.org/register)

### Step 1 — Clone & Install

```bash
git clone https://github.com/Bhawnadhaka/Sentiment_Analysis.git
cd Sentiment_Analysis

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set: NEWS_API_KEY=your_api_key_here
```

### Step 2 — Start Infrastructure

```bash
docker-compose -f docker/docker-compose.yml up -d
# Starts: Kafka, Zookeeper, Redis, Prometheus, Grafana
```

### Step 3 — Train the Model

```bash
python -m src.models.train --model bow
# Saves checkpoint to: models/best_bow_model.pth
# Metrics logged to MLflow (local: mlruns/)
```

### Step 4 — Start the FastAPI Server

```bash
python src/api/main.py
# API available at:  http://localhost:8000
# Swagger UI at:     http://localhost:8000/docs
# Prometheus at:     http://localhost:8000/metrics
```

### Step 5 — Run the Streaming Pipeline

```bash
# Terminal 1: Consumer (waits for Kafka messages)
python run_pipeline.py consumer

# Terminal 2: Producer (fetches news → sends to Kafka)
python run_pipeline.py producer --topics technology AI startups --articles 50
```

### Step 6 — Open the Streamlit Dashboard

```bash
streamlit run app.py
# Dashboard: http://localhost:8501
```

### Step 7 — Generate Monitoring Reports

```bash
python run_pipeline.py dashboard
# HTML reports saved to: monitoring/evidently_reports/
```

### Access All Services

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI (Swagger UI) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

---

## 12. Configuration Files

| File | Key Settings |
|---|---|
| `configs/model_config.yaml` | `type` (distilbert/lstm/bow), `num_classes`, `dropout`, `vocab_size`, `embedding_dim`, `hidden_dim` |
| `configs/training_config.yaml` | `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `max_length`, `early_stopping_patience` |
| `configs/deployment_config.yaml` | API host/port, model path, Kafka settings, Redis TTL |
| `.env` | `NEWS_API_KEY`, `DAGSHUB_USER`, `DAGSHUB_REPO`, `KAFKA_BOOTSTRAP_SERVERS`, `MODEL_PATH`, `MODEL_TYPE` |
| `docker/prometheus.yml` | Prometheus scrape intervals and target endpoints |

---

## 13. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Full system test
python test_monitoring.py
```

| Test File | What It Tests |
|---|---|
| `tests/test_api.py` | FastAPI endpoint responses, error handling, schema validation |
| `tests/test_model.py` | Model creation, forward pass shapes, inference output format |
| `tests/test_data.py` | Dataset loading, tokenization, vocabulary building |
| `tests/test_kafka.py` | Kafka producer/consumer connectivity and message flow |
| `test_monitoring.py` | Evidently report generation and dashboard health |

---

## 14. Key Design Decisions

### BoW Deployed, Not DistilBERT

The lightweight BoW model (76.45% accuracy) was chosen for the deployed API because it is orders of magnitude faster (~50 ms vs ~150 ms) and requires far less memory. For real-time streaming with hundreds of articles per hour, this is the practical choice. DistilBERT is available in the codebase for scenarios where accuracy is prioritised over latency.

### Kafka Over Direct HTTP

Using Kafka decouples the news ingestion rate from the inference rate. The producer can burst many articles without overloading the model server. Consumer groups also allow horizontal scaling — multiple consumers can share the same topic. It also adds durability: if the consumer crashes, it resumes from its last committed offset.

### Redis Caching

Real news tends to repeat — the same headline is often reposted across multiple sources. A 1-hour TTL cache with MD5 hashing avoids redundant model inference and reduces CPU pressure. Measured cache hit rate: ~70%.

### SQLite for Local Dev, PostgreSQL for Production

SQLite requires zero infrastructure and still supports everything needed for development. The docker-compose file also provisions a PostgreSQL container (`postgres:15-alpine`) for production-grade deployments.

### Evidently Over Custom Drift Code

Evidently provides battle-tested statistical tests (PSI, KS, Chi-squared) for drift detection with interactive HTML output and zero boilerplate. Building equivalent functionality from scratch would require significantly more code and maintenance.

### MLflow + DagsHub

DagsHub provides a completely free alternative to a self-hosted MLflow Server + S3 storage for students and solo developers, while being fully compatible with the standard `mlflow` Python API. No infrastructure needed.

---

## 15. Achieved Metrics

| Metric | Value |
|---|---|
| Model Accuracy | 76.45% |
| F1 Score | 76.68% |
| Precision | 76.78% |
| Recall | 76.45% |
| API Latency (uncached) | ~50 ms |
| API Latency (Redis cached) | ~5 ms |
| Cache Hit Rate | ~70% |
| Kafka Streaming | Real-time (bounded by NewsAPI free tier: 100 req/day) |

---

*Built with PyTorch · FastAPI · Apache Kafka · Redis · Prometheus · Grafana · Evidently AI · MLflow · Streamlit · Docker*
