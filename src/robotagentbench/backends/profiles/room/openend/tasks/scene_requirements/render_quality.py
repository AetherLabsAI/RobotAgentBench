"""Image-quality failures are per-view warnings, not renderer failures."""
import hashlib


def record_view(raw, view, output):
    import numpy as np
    from PIL import Image
    pixels = np.asarray(raw)
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        pixels = pixels[:, :, :3]
    shape = list(pixels.shape)
    numeric = np.issubdtype(pixels.dtype, np.number)
    finite = bool(np.isfinite(pixels).all()) if pixels.size and numeric else False
    std = float(pixels.std()) if pixels.size and finite else None
    reason = None
    if shape != [800, 1280, 3]:
        reason = '图片尺寸异常'
    elif not finite:
        reason = '图片包含无效像素'
    elif pixels.dtype != np.uint8:
        reason = '图片像素格式异常'
    elif std < 1:
        reason = '画面接近纯色，可能存在取景或渲染问题'
    if reason:
        warning = dict(name=view['name'], label=view.get('label', view['name']),
                       reason=reason, shape=shape, pixel_std=std)
        # Retain suspicious pixels for diagnosis, but never present them as valid.
        if shape == [800, 1280, 3] and finite and pixels.dtype == np.uint8:
            path = output / ('diagnostic_' + view['name'] + '.png')
            Image.fromarray(pixels).save(path)
            warning['diagnostic_file'] = path.name
        return None, warning
    path = output / (view['name'] + '.png')
    Image.fromarray(pixels).save(path)
    return dict(view, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()), None


def completion_fields(images, warnings, total, attempted):
    if not images:
        raise ValueError('没有可用的渲染图片；场景与诊断记录已保留')
    return dict(warnings=warnings, total_views=total, attempted_views=attempted,
                valid_images=len(images),
                status='completed_with_warnings' if warnings else 'complete')
