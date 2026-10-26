"""User-approved lossless archiving of exactly four FINISHED previews."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
NAMES = ('native_mechanical_002', 'native_mechanical_004', 'native_mechanical_005', 'native_mechanical_006')


def archive(name):
    if name not in NAMES:
        raise ValueError('Not one of the explicitly approved finished previews')
    folder = HERE / name
    frames = folder / 'frames'
    if folder.is_symlink() or frames.is_symlink() or folder.resolve().parent != HERE:
        raise ValueError('Refuse symlink or out-of-task cleanup')
    report = json.loads((folder / 'report.json').read_text())
    if report.get('native_shutdown_complete') is not True:
        raise RuntimeError('Preview is not fully shut down')
    files = sorted(frames.iterdir())
    if not files or any(p.is_symlink() or not p.is_file() or p.suffix != '.png' for p in files):
        raise ValueError('Expected nonempty ordinary PNG-only frame directory')
    dest = folder / 'frames.tar.xz'
    if not dest.exists():
        subprocess.run(['tar', '-cJf', str(dest), '-C', str(folder), 'frames'], check=True)
    expected = {'frames/' + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(dest, 'r:xz') as tf:
        members = tf.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError('Unexpected links/special members in archive')
        actual = {m.name: hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
    if actual != expected:
        raise RuntimeError('Lossless archive member/hash mismatch: NO files removed')
    receipt = dict(archive=str(dest), archive_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                   file_sha256=expected, file_count=len(files), total_original_bytes=sum(p.stat().st_size for p in files),
                   archive_bytes=dest.stat().st_size, all_members_verified_restorable=True,
                   videos_scenes_logs_untouched=True, removed_verified_scattered_frames=False)
    path = folder / 'frames_archive_receipt.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    # Exact validated per-file targets; never a recursive or glob deletion.
    for p in files:
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected['frames/' + p.name]:
            raise RuntimeError('Frame changed since verification; stop deletion')
        p.unlink()
    receipt['removed_verified_scattered_frames'] = True
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:receipt[k] for k in ('archive','file_count','total_original_bytes','archive_bytes',
        'all_members_verified_restorable','removed_verified_scattered_frames')}), flush=True)


if __name__ == '__main__':
    for name in NAMES:
        archive(name)
