# 🚀 MLOps Sentiment Analysis Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLOps](https://img.shields.io/badge/MLOps-Production%20Ready-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Production-ready MLOps pipeline for real-time sentiment analysis using 100% FREE tools**

End-to-end ML system with streaming, monitoring, and deployment

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Deployment](#-deployment)

</div>

---

## ✨ Features

**MLOps Pipeline**
- 🔄 Data versioning with DVC
- 📊 Experiment tracking with MLflow
- 🚀 CI/CD with GitHub Actions
- 📦 Docker containerization
- ☁️ Cloud deployment ready (Fly.io)

**Machine Learning**
- 🤖 Multiple models: BoW (76.45%), LSTM, DistilBERT
- ⚡ FastAPI REST API with OpenAPI docs
- 💾 Redis caching (10x speedup)
- 🎯 Real-time predictions

**Streaming & Monitoring**
- 📡 Apache Kafka for real-time processing
- 📈 Prometheus + Grafana dashboards
- 🔍 Evidently AI model monitoring
- 🎨 Streamlit web dashboard
- 📊 Data drift detection

---

## 🛠️ Tech Stack

**ML**: PyTorch, Transformers, Scikit-learn  
**API**: FastAPI, Uvicorn, Redis  
**Streaming**: Apache Kafka, Zookeeper  
**Monitoring**: Prometheus, Grafana, Evidently AI  
**MLOps**: MLflow, DVC, DagsHub  
**DevOps**: Docker, GitHub Actions, Pre-commit  
**Deployment**: Fly.io (free tier)  
**UI**: Streamlit

All tools are **100% free** for learning and development!

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ News API    │─────▶│    Kafka     │─────▶│  FastAPI    │
│ (Producer)  │      │  (Streaming) │      │  (API+ML)   │
└─────────────┘      └──────────────┘      └──────┬──────┘
                                                   │
                     ┌──────────────┐             │
                     │  Prometheus  │◀────────────┤
                     │  (Metrics)   │             │
                     └──────┬───────┘             │
                            │                     │
                     ┌──────▼───────┐      ┌──────▼──────┐
                     │   Grafana    │      │   Redis     │
                     │ (Dashboards) │      │  (Cache)    │
                     └──────────────┘      └─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Docker services
docker-compose -f docker/docker-compose.yml up -d

# 4. Run API
python src/api/main.py

# 5. Open Streamlit Dashboard
streamlit run app.py
```

### Quick Test
```bash
# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'

# Run complete system test
python test_monitoring.py
```

---

## 📊 Model Performance

| Model | Accuracy | F1 Score | Speed | Training Time |
|-------|----------|----------|-------|---------------|
| **BoW** (current) | 76.45% | 0.76 | 50ms | 30 min |
| LSTM | ~80% | ~0.80 | 100ms | 3-4 hours |
| DistilBERT | ~85% | ~0.85 | 150ms | 8-10 hours |

**Current deployed model**: Bag-of-Words (BoW) with 76.45% accuracy

---

## 📁 Project Structure

```
├── src/
│   ├── api/              # FastAPI application
│   ├── models/           # ML model code
│   ├── data/             # Data processing
│   ├── monitoring/       # Monitoring tools
│   └── streaming/        # Kafka producers/consumers
├── data/
│   ├── raw/              # Raw dataset (1.6M tweets)
│   └── processed/        # Preprocessed data
├── docker/               # Docker configurations
├── tests/                # Unit tests
├── configs/              # YAML configurations
├── models/               # Trained models
├── monitoring/           # Evidently AI reports
├── app.py                # Streamlit dashboard
├── run_pipeline.py       # Pipeline orchestration
└── test_monitoring.py    # System testing
```

---

## 🔧 Configuration

### Environment Variables
Create `.env` file:
```bash
NEWS_API_KEY=your_news_api_key
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
REDIS_HOST=localhost
REDIS_PORT=6379
MODEL_TYPE=bow
```

### Docker Services
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Services:
# - Kafka (port 9092)
# - Zookeeper (port 2181)
# - Redis (port 6379)
# - Prometheus (port 9090)
# - Grafana (port 3000, admin/admin)
# - PostgreSQL (port 5432)
```

---

## 📈 Monitoring

### Dashboards
- **Streamlit**: http://localhost:8501 - Main dashboard
- **FastAPI Docs**: http://localhost:8000/docs - API documentation
- **Prometheus**: http://localhost:9090 - Metrics
- **Grafana**: http://localhost:3000 - Visualizations
- **MLflow**: `mlflow ui` → http://localhost:5000 - Experiments

### Key Metrics
- API response time: ~50ms (uncached), ~5ms (cached)
- Cache hit rate: ~70%
- Predictions stored: SQLite database
- Real-time streaming: Kafka processing

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
1. **Testing**: Pytest, flake8, black, mypy
2. **Data Validation**: Schema checks
3. **Model Training**: Automated retraining
4. **Docker Build**: Container builds + security scan
5. **Integration Tests**: API health checks
6. **Deployment**: Fly.io production deploy

Configure secrets in GitHub:
- `FLY_API_TOKEN`
- `NEWS_API_KEY`
- `MLFLOW_TRACKING_URI`

---

## ☁️ Deployment

### Fly.io (Free Tier)
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy
flyctl launch --name sentiment-analysis
flyctl secrets set NEWS_API_KEY=your_key
flyctl deploy

# Monitor
flyctl status
flyctl logs
```

Free tier includes: 3 shared VMs, 3GB storage, 160GB bandwidth

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test specific component
pytest tests/test_api.py

# Complete system test
python test_monitoring.py
```

---

## 📊 Data Pipeline

### Dataset
- **Source**: Kaggle Sentiment140 (1.6M tweets)
- **Split**: 80% train, 10% validation, 10% test
- **Preprocessing**: Cleaning, tokenization, stopword removal

### Real-time Streaming
```bash
# Start news producer
python run_pipeline.py producer --topics technology AI --articles 20

# Start consumer
python run_pipeline.py consumer

python run_pipeline.py dashboard
```

---

## 🛠️ Development

### Pre-commit Hooks
```bash
# Setup
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **Bandit**: Security scanning
- **mypy**: Type checking

---

## 📚 What You'll Learn

✅ End-to-end MLOps pipeline  
✅ Real-time streaming with Kafka  
✅ Model monitoring & drift detection  
✅ CI/CD for ML systems  
✅ Docker containerization  
✅ REST API development  
✅ Experiment tracking with MLflow  
✅ Production deployment

---

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

---

## 📝 License

MIT License - Free for learning and portfolios!

---

## 🙏 Acknowledgments

Built with: PyTorch • FastAPI • Kafka • MLflow • Prometheus • Docker

---

<div align="center">

**Made with ❤️ for MLOps learning**

⭐ Star this repo if helpful!

</div>

<div align="center">

⭐ **Star this repo if it helped you learn MLOps!** ⭐

</div>
