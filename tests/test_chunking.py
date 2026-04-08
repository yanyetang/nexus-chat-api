"""Tests for product text chunking utility."""

from app.utils.chunking import product_to_chunk

_SAMPLE_PRODUCT = {
    "id": "prod-1",
    "title": "Unisex Staple T-Shirt",
    "description": "High-quality 100% cotton tee",
    "brand": "Bella+Canvas",
    "model": "BSR-001",
    "category": {"title": "T-Shirts"},
    "tags": ["bestseller", "unisex"],
    "variants": [
        {
            "title": "White Medium",
            "price": "19.99",
            "size": "M",
            "color": "White",
            "sku": "TSR-WHT-M",
            "inventory": 145,
        }
    ],
}


def test_chunk_contains_title():
    chunk = product_to_chunk(_SAMPLE_PRODUCT)
    assert "Unisex Staple T-Shirt" in chunk


def test_chunk_contains_category():
    chunk = product_to_chunk(_SAMPLE_PRODUCT)
    assert "T-Shirts" in chunk


def test_chunk_contains_variant_sku():
    chunk = product_to_chunk(_SAMPLE_PRODUCT)
    assert "TSR-WHT-M" in chunk


def test_chunk_contains_tags():
    chunk = product_to_chunk(_SAMPLE_PRODUCT)
    assert "bestseller" in chunk


def test_chunk_handles_missing_fields():
    chunk = product_to_chunk({"title": "Minimal Product"})
    assert "Minimal Product" in chunk
    assert "No variants" in chunk


def test_chunk_handles_empty_product():
    chunk = product_to_chunk({})
    assert isinstance(chunk, str)
