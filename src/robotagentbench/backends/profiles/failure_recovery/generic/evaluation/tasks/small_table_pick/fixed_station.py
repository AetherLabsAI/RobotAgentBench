"""Approved reset pose only; no runtime attachment, control or physics edits."""
import json
from .definitions import TASK, SUPPORT


def definition():
    value = json.loads((TASK/'references'/'fixed_front_station.json').read_text())
    if value['support'] != SUPPORT or value['independent_base_randomization'] is not False:
        raise ValueError('Fixed small-table station contract mismatch')
    return value


def robot_world(table_world, current_robot_world):
    """USD row-vector convention: robot_in_table * table_in_world.

    Preserve robot asset scale. Table translation/yaw carries the base pose;
    no independent robot sampling or derivation from the old diagonal station.
    """
    from pxr import Gf
    value = definition()
    relative = Gf.Matrix4d(value['robot_in_table_rigid_frame'])
    scale = Gf.Matrix4d(1).SetScale(Gf.Transform(current_robot_world).GetScale())
    return scale * relative * table_world.RemoveScaleShear()
