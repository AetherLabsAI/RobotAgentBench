"""Exact original image build inputs. A rebuild need not have the same image ID."""

def main(argv=None):
    import argparse,hashlib,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path)
    a=p.parse_args(argv)
    if hashlib.sha256(DOCKERFILE).hexdigest()!=DOCKERFILE_SHA256 or hashlib.sha256(DEPENDENCY_LOCK).hexdigest()!=LOCK_SHA256:
        raise ValueError('Original build inputs differ')
    if a.export is not None:
        root=a.export.resolve();root.mkdir(parents=True,exist_ok=False)
        for name,data in (('Dockerfile.motion',DOCKERFILE),('requirements.lock',DEPENDENCY_LOCK)):
            with (root/name).open('xb')as stream:stream.write(data)
    print(json.dumps(dict(server_image=SERVER_IMAGE,recipe_sha256=DOCKERFILE_SHA256,
        lock_sha256=LOCK_SHA256,built=False,exact_archive_bundled=False)))
    return 0

SERVER_IMAGE = "sha256:4ccf7e950d48f9b5477292bbc123b676fdbad039c6d51573d969008d8ffd5dc6"
LAUNCHER_SHA256 = "40dcd4173e7da2885adf0d96178386e952b2a4d2d1b1212b6e2cf0dc83ba750a"
DOCKERFILE_SHA256 = "bc6968eaa7d9d239fecd473a71e671c63e471cc27c4987ff350b53787fda111c"
LOCK_SHA256 = "c74fdf44032127126f4467fad74db8e29eacb086415858243652e062fde5a822"
DOCKERFILE = "FROM python@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84\nRUN pip install --no-cache-dir numpy==2.2.6 mujoco==3.13.0 mink==1.3.0 quadprog==0.1.13 \\\n    && pip check \\\n    && pip freeze > /opt/motion-requirements.lock\nENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n".encode("utf-8")
DEPENDENCY_LOCK = "absl-py==2.5.1\ndaqp==0.10.3\netils==1.14.0\nfsspec==2026.9.0\nglfw==2.11.0\nmink==1.3.0\nmujoco==3.13.0\nnumpy==2.2.6\npackaging==26.3\nPyOpenGL==3.1.10\nqpsolvers==4.13.0\nquadprog==0.1.13\nscipy==1.17.1\ntyping_extensions==4.16.0\nzipp==4.1.1\n".encode("utf-8")

if __name__=='__main__':raise SystemExit(main())

