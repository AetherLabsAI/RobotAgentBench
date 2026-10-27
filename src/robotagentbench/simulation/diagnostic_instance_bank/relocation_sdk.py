"""Compatibility read of a proven existing native offset; never set physics.

Installed beside the original task as ``_sdk_effective_contact.py``. The
original arm-plane formula is unchanged: forearm plus an upper-arm group only
when that original asset actually contains one. No default or estimate exists.
"""
import hashlib
import json
import math
from pathlib import Path

MEASUREMENT_SHA256='c4f44759d7c82e2a57d47aed4ba5051071192098ef04f733339a28f197b07388'
SDK_FILES={
 'isaacsim-5.1.0.0.dist-info/METADATA':'62886aa8eed75cdd927a5865055e4cf957b271a623f9c1c74e1cbcb51a84a7b9',
 'isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/generatedSchema.usda':'d2dd1873e165d280ac302f2b2e6e95cc5c4c1f72580478492990ea0f711e00f8',
 'isaacsim/extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physics/tensors/impl/api.py':'5dd16f8a37eccc94ac82338d6c1127e785cccf761cae1f6f18ef03d55b0f325f',
}


def validated_margin(receipt,groups,*,expected_sdk,expected_shape):
    if (receipt.get('status')!='measured' or receipt.get('original_initial_accepted')is not True
        or receipt.get('native_state_and_clock_unchanged')is not True
        or receipt.get('physics_attributes_written')is not False
        or receipt.get('shapes_created_or_enabled')is not False
        or receipt.get('control_actions_sent')!=0 or receipt.get('trigger_forced')is not False
        or receipt.get('sdk_identity')!=expected_sdk or receipt.get('shape_identity')!=expected_shape):
        raise ValueError('Unchanged original native measurement and provenance required')
    physics=receipt['scene_physics_identity']
    if physics.get('meters_per_unit')!=1 or physics.get('physics_hz')!=60 or physics.get('gpu_dynamics')is not False:
        raise ValueError('Original metre/CPU/60Hz identity required')
    rows=receipt['rows']
    if set(rows)!=set(groups) or 'forearm.R'not in groups or 'hand.R'not in groups:
        raise ValueError('Complete original asset collision groups must match measured bodies')
    # Original initialization.reviewed_orientation:55. Do not invent an upper
    # arm for a cropped asset, and do not ignore it if the asset contains one.
    consumers=['forearm.R']+(['upper_arm.R']if 'upper_arm.R'in groups else[])
    offsets=[]
    for name in consumers:
        row=rows[name];body='/World/HumanInterference/Bodies/'+name.replace('.','_')
        if row.get('body')!=body or row.get('collider')!=body+'/Collision':
            raise ValueError('Actual consumed native collider identity changed')
        value=row['contact_offset_stage_units'];rest=row['rest_offset_stage_units']
        if (type(value)is not float or not math.isfinite(value) or value<0
            or type(rest)is not float or not math.isfinite(rest) or rest>value):
            raise ValueError('Finite measured consumed native shape offset required')
        offsets.append(value)
    if len({v.hex()for v in offsets})!=1:
        raise ValueError('Consumed arm colliders do not have bit-identical effective offsets')
    return offsets[0]


def read_margin(asset):
    """Read verified measurement for exactly this unchanged asset and SDK."""
    here=Path(__file__).resolve().parent
    config=json.loads((here/'sdk_offset_input.json').read_text())
    raw=(here/'native_effective_offsets.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MEASUREMENT_SHA256:
        raise ValueError('Original native offset evidence changed')
    receipt=json.loads(raw);site=Path(config['sdk_site'])
    for relative,expected in SDK_FILES.items():
        if hashlib.sha256((site/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('Native SDK differs from the measured profile: '+relative)
    shape=dict(receipt['shape_identity'])
    if hashlib.sha256(asset.path.read_bytes()).hexdigest()!=shape['expected_asset_sha256']:
        raise ValueError('Original cropped human asset differs from measurement')
    if hashlib.sha256((here/'human_asset.py').read_bytes()).hexdigest()!=shape['human_asset_module_sha256']:
        raise ValueError('Original native human collider construction differs from measurement')
    rig=here/'static_short_hand_011/cropped_rig.json'
    if hashlib.sha256(rig.read_bytes()).hexdigest()!=shape['expected_rig_sha256']:
        raise ValueError('Original cropped human rig differs from measurement')
    return validated_margin(receipt,asset.collision_groups(),
        expected_sdk=config['expected_sdk'],expected_shape=config['expected_shape'])
