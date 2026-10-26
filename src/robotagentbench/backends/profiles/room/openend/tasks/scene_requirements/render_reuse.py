"""Reuse hash-verified images from the identical static scene for targeted repairs."""
import hashlib,json,re,shutil


def reuse_images(folder,output,source,selected,views):
    receipt=json.loads((folder/source/'manifest.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if receipt['scene_sha256']!=digest(folder/'room.usda'):
        raise ValueError('Cannot reuse images from a different scene')
    names={v['name'] for v in views}
    if not selected or not set(selected)<=names:
        raise ValueError('Unknown or empty repair view selection')
    retained=[v for v in receipt['images'] if v['name'] not in selected]
    if {v['name'] for v in retained}|set(selected)!=names:
        raise ValueError('Repair selection does not cover all missing views')
    for item in retained:
        if not re.fullmatch(r'(overview|living|kitchen|area_[a-z_]+)\.png',item['file']):
            raise ValueError('Invalid retained image filename')
        src=folder/source/item['file']
        if digest(src)!=item['sha256']:raise ValueError('Retained image hash mismatch')
        shutil.copy2(src,output/item['file'])
    return retained,dict(directory=source,manifest_sha256=digest(folder/source/'manifest.json'),
                         retained_images=len(retained),repaired_views=selected)
