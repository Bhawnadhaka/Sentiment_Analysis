# Makefile for common commands

.PHONY: help install install-dev setup test lint format train api docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make setup         - Full project setup"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make train         - Train model"
	@echo "  make api           - Run API server"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make clean         - Clean generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
	git init
	dvc init
	@echo "✓ Setup complete! Next steps:"
	@echo "1. Download data: python -m data.scripts.download_kaggle"
	@echo "2. Preprocess data: python -m data.scripts.preprocess"
	@echo "3. Train model: make train"

test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

lint:
	flake8 src/
	mypy src/ --ignore-missing-imports

format:
	black src/
	isort src/

train:
	python -m src.models.train --model distilbert

train-lstm:
	python -m src.models.train --model lstm

train-bow:
	python -m src.models.train --model bow

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✓ Services started:"
	@echo "  - API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo "  - Kafka: localhost:9092"
	@echo "  - PostgreSQL: localhost:5432"

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

data-download:
	python -m data.scripts.download_kaggle

data-preprocess:
	python -m data.scripts.preprocess

dvc-push:
	dvc add data/raw data/processed
	dvc push

dvc-pull:
	dvc pull

mlflow-ui:
	mlflow ui --port 5000

monitoring-report:
	python -m src.monitoring.drift_detector
