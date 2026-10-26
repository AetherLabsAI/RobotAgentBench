"""Exact catalogue-product uniqueness throughout this task's room."""


def background_pool(pool, target_product):
    """Initialization helper: other product types are still allowed."""
    return [product for product in pool if product != target_product]


def require_unique_target(manifest, product, target_path=None):
    # Count all sources and roles, not merely requested shelf products.
    matches=[o for o in manifest['objects'] if o.get('id')==product]
    if len(matches)!=1:
        raise ValueError('Target product must occur exactly once in the entire room: '
                         +product+' (found '+str(len(matches))+')')
    target=matches[0]
    if not isinstance(target.get('path'),str) or not target['path'].startswith('/World/'):
        raise ValueError('Unique target needs an exact scene path')
    if target_path is not None and target['path']!=target_path:
        raise ValueError('Unique product and registered target instance disagree')
    return target
