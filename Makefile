.PHONY: help install dev test lint type-check format clean

help:
	@echo "Available commands:"
	@echo "  make dev          - Start development server with auto-reload"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Check code style with ruff"
	@echo "  make type-check   - Type check with pyright"
	@echo "  make format       - Format code with ruff"
	@echo "  make install      - Install all dependencies"
	@echo "  make clean        - Remove build artifacts and cache"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

dev:
	.venv/bin/uvicorn app.main:app --reload

test:
	.venv/bin/pytest tests/ -q

lint:
	.venv/bin/ruff check app

type-check:
	.venv/bin/pyright --pythonpath .venv/bin/python app

format:
	.venv/bin/ruff format app

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache .pyright
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
