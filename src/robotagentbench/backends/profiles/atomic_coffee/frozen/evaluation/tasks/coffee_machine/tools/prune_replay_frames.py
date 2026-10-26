"""Delete the per-step PNG frames of finished replay jobs; keep the encoded videos.

User rule 2026-09-14: after a replay job is complete (CPU audit + three-camera videos exist), only the videos are
kept. Step 0 images are kept. A FRAMES-REMOVED.txt marker records what was deleted. Never touches probes, audits,
reports, videos or anything outside the job directory. Same job layout and code as the other task families.
"""
import argparse
import json
import time
from pathlib import Path

CAMS = ('head', 'left', 'right')


def prunable(job):
    manifest = job/'videos'/'manifest.json'
    if not manifest.exists() or not (job/'run'/'physical-audit.json').exists(): return False
    videos = {v['camera']: job/'videos'/v['file'] for v in json.loads(manifest.read_text())['videos']}
    return all(videos.get(c) and videos[c].exists() and videos[c].stat().st_size > 0 for c in CAMS)


def prune(job, *, dry=False):
    job = Path(job)
    if not prunable(job): return None
    dirs = [job/'run'/'episode'] + sorted(p for p in (job/'episodes').glob('*') if p.is_dir()) if (job/'episodes').exists() else [job/'run'/'episode']
    removed = 0; freed = 0
    for d in dirs:
        for f in d.glob('step*.png'):
            if f.name.startswith('step0000_'): continue
            freed += f.stat().st_size; removed += 1
            if not dry: f.unlink()
    if removed and not dry:
        (job/'FRAMES-REMOVED.txt').write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} removed {removed} step PNG frames ({freed/1e9:.2f} GB) after completion; videos/*.mp4, run/physical-audit.json, run/report.json and step0000 images are kept (user rule 2026-09-14: keep videos only).\n")
    return dict(job=str(job), removed=removed, freed_gb=round(freed/1e9, 2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('paths', nargs='+', type=Path, help='job directories or batch roots')
    p.add_argument('--dry', action='store_true'); a = p.parse_args()
    total = 0
    for root in a.paths:
        jobs = [root] if (root/'videos').exists() else sorted(d for d in root.glob('*/*') if (d/'videos').exists())
        for job in jobs:
            r = prune(job, dry=a.dry)
            if r and r['removed']: total += r['freed_gb']; print(json.dumps(r))
    print(json.dumps(dict(total_freed_gb=round(total, 2), dry=a.dry)))

if __name__ == '__main__': main()
