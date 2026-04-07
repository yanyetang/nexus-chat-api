"""Shared test fixtures and configuration."""

import pytest

# Provide placeholder env vars so the app can be imported without a real .env
_TEST_ENV = {
    "DATABASE_URL": "postgresql://placeholder:placeholder@localhost:5432/placeholder",
    "SUPPLIER_API_BASE_URL": "https://placeholder.example.com",
    "COHERE_API_KEY": "placeholder",
    "OPENROUTER_API_KEY": "placeholder",
}


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    for key, value in _TEST_ENV.items():
        monkeypatch.setenv(key, value)
