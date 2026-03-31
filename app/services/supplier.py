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
