import httpx

from app.config import get_settings
from app.exceptions import ExternalServiceError


class SupplierService:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.supplier_api_base_url.rstrip("/")

    async def fetch_catalog_export(self) -> list[dict]:
        url = f"{self._base_url}/catalog/export"
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise ExternalServiceError("Supplier API unavailable") from exc

        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return data["data"]
        if isinstance(data, list):
            return data
        return []

    async def fetch_products_by_ids(self, product_ids: list[str]) -> dict[str, dict]:
        ids = [str(product_id) for product_id in product_ids if product_id]
        if not ids:
            return {}

        # Prefer supplier-side filtering when available; fall back to local filtering on full export.
        url = f"{self._base_url}/catalog/export"
        params = {"ids": ",".join(ids)}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError:
            products = await self.fetch_catalog_export()
            return {
                str(item.get("id")): item
                for item in products
                if item.get("id") is not None and str(item.get("id")) in ids
            }

        products: list[dict]
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            products = payload["data"]
        elif isinstance(payload, list):
            products = payload
        else:
            products = []

        if not products:
            products = await self.fetch_catalog_export()
            return {
                str(item.get("id")): item
                for item in products
                if item.get("id") is not None and str(item.get("id")) in ids
            }

        return {
            str(item.get("id")): item
            for item in products
            if item.get("id") is not None and str(item.get("id")) in ids
        }
