# 🚀 Sentiment Analysis MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

**Real-time sentiment analysis with streaming, caching, and monitoring**

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api-endpoints)

</div>

---

## ✨ Features

- 🤖 Sentiment analysis model (76.45% accuracy)
- ⚡ FastAPI REST API 
- 📡 Apache Kafka streaming
- 💾 Redis caching
- 📈 Prometheus + Grafana monitoring
- 🎨 Streamlit dashboard
- 📊 Evidently AI drift detection
- 🐳 Docker containerization

---

## 🛠️ Tech Stack

**ML**: PyTorch, Scikit-learn  
**API**: FastAPI, Redis  
**Streaming**: Apache Kafka  
**Monitoring**: Prometheus, Grafana, Evidently AI  
**Tools**: MLflow, Docker, Streamlit

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Bhawnadhaka/Sentiment_Analysis.git
cd Sentiment_Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start services
docker-compose -f docker/docker-compose.yml up -d

# 4. Run API
python src/api/main.py

# 5. Open dashboard
streamlit run app.py
```

**Access:**
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## 📊 Model Performance

| Model | Accuracy | Speed |
|-------|----------|-------|
| **BoW** (deployed) | 76.45% | 50ms |
| LSTM | ~80% | 100ms |
| DistilBERT | ~85% | 150ms |

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model info |
| `/metrics` | GET | Prometheus metrics |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

---

## 📈 Monitoring

**Dashboards:**
- Streamlit: Real-time predictions & analytics
- Prometheus: Metrics collection
- Grafana: Visualizations
- Evidently: Model drift detection

**Key Metrics:**
- Response time: ~50ms (uncached), ~5ms (cached)
- Cache hit rate: ~70%
- Kafka streaming: Real-time processing

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# System test
python test_monitoring.py
```

---

## 📝 License

MIT License

---

<div align="center">

**Built with PyTorch • FastAPI • Kafka • Docker**

</div>
