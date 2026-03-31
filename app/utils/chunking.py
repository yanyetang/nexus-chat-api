def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def product_to_chunk(product: dict) -> str:
    title = _safe_text(product.get("title"))
    description = _safe_text(product.get("description"))
    brand = _safe_text(product.get("brand"))
    model = _safe_text(product.get("model"))

    category = product.get("category") or {}
    category_title = _safe_text(category.get("title"))

    tags = product.get("tags") or []
    tag_text = ", ".join(str(tag) for tag in tags if str(tag).strip())

    variant_lines: list[str] = []
    for variant in product.get("variants") or []:
        variant_lines.append(
            "- {title}: ${price}, Size: {size}, Color: {color}, SKU: {sku}, Stock: {stock}".format(
                title=_safe_text(variant.get("title")) or "Unknown",
                price=_safe_text(variant.get("price")) or "0",
                size=_safe_text(variant.get("size")) or "N/A",
                color=_safe_text(variant.get("color")) or "N/A",
                sku=_safe_text(variant.get("sku")) or "N/A",
                stock=_safe_text(variant.get("inventory"))
                or _safe_text(variant.get("inventoryQty"))
                or "0",
            )
        )

    variants_block = "\n".join(variant_lines) if variant_lines else "- No variants"

    return (
        f"Product: {title}\n"
        f"Brand: {brand} | Model: {model} | Category: {category_title}\n"
        f"Description: {description}\n"
        f"Tags: {tag_text}\n"
        f"Variants:\n{variants_block}"
    )
