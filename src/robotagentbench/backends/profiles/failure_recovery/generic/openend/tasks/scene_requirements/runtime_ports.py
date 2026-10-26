"""Scene-owned source adapters. No reset, action, or physics parameter writes."""
import json
from dataclasses import asdict
import sys
from pathlib import Path
import numpy as np
# The existing component package has sibling imports. Resolve its deployed
# location, not a historical L1/L2 path and not a caller's working directory.
COMPONENT_ROOT=Path(__file__).resolve().parents[3]/'layout_task_activation_v1/task_scene_registration_v1'
if str(COMPONENT_ROOT) not in sys.path:sys.path.insert(0,str(COMPONENT_ROOT))
from openend.tasks.glasses_transfer.port import GlassesPort as BaseGlassesPort
from openend.tasks.wallet_transfer.port import WalletPort as BaseWalletPort
from layout_task_activation_v1.task_scene_registration_v1.fridge_to_microwave_v1.room_port import (
    FridgeMicrowavePort as BaseFridgePort, normalize_parameters)


class FridgePort(BaseFridgePort):
    supported_physics_hz=(60,240)
    initial_workspace = 'fridge'
    required_workspaces = ('fridge', 'microwave')
    validate_parameters = staticmethod(normalize_parameters)

    @staticmethod
    def resolve_object(parameters, manifest):
        variant = normalize_parameters(parameters)
        rows = [o for o in manifest['objects'] if o['id'] == variant]
        if len(rows) != 1: raise ValueError('Exactly one selected fridge payload required')
        return rows[0]

    @property
    def workspace_id(self):
        return 'fridge' if self.composite.phase == 'refrigerator' else 'microwave'

    def selection_geometry(self):
        lo, hi = self.scene.activation[self.composite.phase]
        return (tuple((lo[k]+hi[k])/2 for k in (0,1)),
                ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])))

    def initialize(self, clock, view):
        from openend.tasks.holding_follow_v1.fridge import FollowFridgeSource
        self.clock=clock
        self.source=FollowFridgeSource(self.scene,clock,view)
        from openend.tasks.meal_tray.nonmotion_closeout_v1.contact_stream import RoomPutContacts
        self.source.contacts = RoomPutContacts(self.source.contacts, self.output)

    def after_physics(self, helper_report):
        # NativeSource already checks the actual 60/240 Hz delta and computes
        # contact forces from impulse / actual dt. Keep the original component
        # untouched; this room adapter owns variable-rate control intervals.
        self.source.after_physics(helper_report,self.composite.phase)
        if self.enabled:self.buffer.append(self.source.latest)

    def observe(self):
        from openend.tasks.meal_tray.physics_process_final_v1.clock_contract import validate_interval
        if not self.enabled:raise RuntimeError('Unselected refrigerator evidence')
        validate_interval(self.buffer,self.clock.hz,self.clock.current_time)
        frame=self.source.frame(self.episode,self.index+1,self.buffer)
        self.composite.observe(frame);self.index+=1;self._summary()
        self.trace.write(json.dumps(dict(facts=asdict(frame),details=self.source.latest['details'],score=self.score))+'\n')
        self.trace.flush();self.discard_control()
        return self.score

    def after_inactive(self, helper_report):
        # No native pose reads/scoring while away. Fresh selected substeps are
        # required on reentry; the original composite pauses unfinished holds.
        now = float(self.clock.current_time)
        if abs(now-self.source.last_time-self.clock.get_physics_dt()) > 1e-6:
            raise RuntimeError('Missing inactive native substep')
        if helper_report.get('events') or helper_report.get('docked_object_count') or helper_report.get('tray_dock_enabled'):
            raise RuntimeError('Assisted physical evidence')
        self.source.last_time = now
        # Keep only the contact stream current so real sleeping support is not
        # lost on return; do not run the full multi-body scoring reader.
        from evaluation.tasks.microwave_put.physical_geometry import PLATE_BODY,rotation
        _,q=self.source.views[PLATE_BODY].get_world_pose()
        self.source.contacts.read(self.source.interface.get_contact_report(),rotation(q).apply([0,0,1]))

    def finalize(self, reason):
        self.discard_control()
        self.composite.finalize()
        return self._summary()


class DrawerSource:
    """Reuse the old 9-1/9-2 collider-above-rim predicate, measured natively."""
    def __init__(self, stage, manifest, *args, **kwargs):
        super().__init__(stage, manifest, *args, **kwargs)
        row, = [o for o in manifest['objects'] if o['id'] == self.asset]
        self.drawer = row.get('drawer_path') if row.get('source') == 'coffee_table.drawer' else None
        if row.get('source') == 'coffee_table.drawer' and not self.drawer:
            raise ValueError('Drawer source requires its actual body identity')
        if self.drawer:
            from evaluation.tasks.find_object.isaac_evidence import body_colliders
            cols = body_colliders(stage, self.drawer)
            walls = [v for name,v in cols.items() if 'Bottom' not in name]
            if not walls: raise ValueError('Missing drawer wall colliders')
            self.drawer_walls = np.vstack(walls)
            self.target_colliders = np.vstack(list(body_colliders(stage,self.geometry.body).values()))
            (self.output/'source_binding.json').write_text(json.dumps(dict(
                source='coffee_table.drawer', drawer=self.drawer,
                pickup_api='9-1' if self.asset=='glasses_case' else '9-2',
                predicate='lowest_target_collider_z > highest_current_drawer_wall_z',
                thresholds_changed=False, full_physical_trajectory_verified=False),indent=2)+'\n')

    def initialize(self, clock, view):
        super().initialize(clock, view)
        if self.drawer:
            self.drawer_view = view.create_rigid_body_view(self.drawer)
            if self.drawer_view.count != 1 or list(self.drawer_view.prim_paths) != [self.drawer]:
                raise ValueError('Drawer native body identity mismatch')

    def after_physics(self, helper_report):
        super().after_physics(helper_report)
        if self.drawer:
            from evaluation.tasks.shelf_pick.geometry import rotation
            t = np.asarray(self.drawer_view.get_transforms()).copy()[0]
            walls = rotation(tuple(t[[6,3,4,5]])).apply(self.drawer_walls)+t[:3]
            pose = self.latest['pose']
            target = rotation(pose.quaternion_wxyz).apply(self.target_colliders)+pose.position_m
            self.latest['source_clear'] = bool(target[:,2].min() > walls[:,2].max())


class GlassesPort(DrawerSource, BaseGlassesPort): pass
class WalletPort(DrawerSource, BaseWalletPort): pass
