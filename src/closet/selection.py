"""Pair-specific curated links: one outfit may have different anchor garments."""

def connection(selection, product_id, outfit_id):
    if selection.get('version', 0) >= 4:
        return selection.get('connections', {}).get(product_id, {}).get(outfit_id, {})
    return selection.get('outfits', {}).get(outfit_id, {})

def allowed_products(selected, garment_id):
    if 'groups' in selected:
        return selected['groups'].get(garment_id, [])
    return selected.get('products', [])
