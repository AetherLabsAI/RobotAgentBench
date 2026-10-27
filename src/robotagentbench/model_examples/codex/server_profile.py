"""Export exact server coding delta for an independently reviewed source view.

This never mutates Runtime, backends, source pins or the running environment.
The original task/scoring/physics sources are not part of this delta.
"""
import argparse,hashlib,json
from pathlib import Path
from .starter_data import DATA

PREFIX='short_tasks/agent_probe_v1/cap_v1/native_tools/'
FILES={
    'sandbox_launcher.py':'40dcd4173e7da2885adf0d96178386e952b2a4d2d1b1212b6e2cf0dc83ba750a',
    'starter/g1_model.py':'e3182ab96230feedb87578a85c3404406192dca7d9e2cdbccc55a4c7005014b4',
    'starter/move_to.py':'a7c4fc5a454a62d292a5e0356f1babb6e09b2f26fabe6c562f1b65248e938756',
    'starter/planned_motion.py':'94ec757a965428d7fdc6973c8412539cc3642fe9e4bd71d6660b19af3268320c',
    'starter/self_collision.py':'378b8d91e578013c76b59f1e84f017c091ef0544581ac798eb2b9c333a610741',
    'starter/robot/robot_arm_kinematics.xml':'80bc15c3465fd6f0c89b7590afc2cb20974b6f1f81b1266ec06cf4b21748d03e',
    'starter/robot/robot_contract.json':'bcb4820bd11391b002180753b5a94d4d34facca79c0462ace047f98e34fa8987',
    'starter/robot/robot_self_primitives.json':'bf5515aaadceb22f5d54abbe5572978650b4b89e9d9001cc006a76e87c2e30dd',
    'starter/UPSTREAM.md':'f463b42628ef978a2d49363e055d76e59afd49a4518d67edd5b14b030425e2cf',
}

def payloads():
    root=Path(__file__).resolve().parent/'original_sources';result={}
    for relative,expected in FILES.items():
        raw=DATA[relative[len('starter/'):]]if relative.startswith('starter/')and relative[len('starter/'):]in DATA else (root/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Original coding resource differs: '+relative)
        result[PREFIX+relative]=raw
    return result

def deployment_payloads(image):
    """Bind a trusted operator's local build without modifying original sources.

    Only the launcher's image literal changes. All isolation flags, robot tools,
    task/scoring sources and original provenance stay intact.
    """
    from .sandbox_build import image_id
    image_id(image)
    data=payloads()
    path=PREFIX+'sandbox_launcher.py'
    original=data[path]
    declaration=b"IMAGE = 'sha256:4ccf7e950d48f9b5477292bbc123b676fdbad039c6d51573d969008d8ffd5dc6'"
    if original.count(declaration)!=1:raise ValueError('Original image declaration differs')
    data[path]=original.replace(declaration,('IMAGE = '+repr(image)).encode())
    return data

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path)
    a=p.parse_args(argv);data=payloads()
    if a.export is not None:
        root=a.export.resolve();root.mkdir(parents=True,exist_ok=False)
        for relative,raw in data.items():
            target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb')as stream:stream.write(raw)
        manifest=dict(schema='robotagentbench.codex_server_source_delta.v1',
            files_sha256={name:hashlib.sha256(raw).hexdigest()for name,raw in data.items()},
            original_launcher_sha256='500024e1a0d67afcd64f48e1c653890bb7ab3b0d42db829cc1ed721a39b64a4a',
            automatic_runtime_application=False,scoring_or_physics_edited=False)
        with (root/'source_delta.json').open('x')as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(files=len(data),verified=True,runtime_modified=False,native_started=False)))
    return 0

if __name__=='__main__':raise SystemExit(main())
