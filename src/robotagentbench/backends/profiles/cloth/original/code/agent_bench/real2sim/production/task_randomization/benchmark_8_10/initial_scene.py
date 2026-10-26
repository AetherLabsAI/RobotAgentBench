"""Independent supported-canvas scene setup for Sheet 8.10.

Geometry is built from the material grid and measured basket, then settled with
no attachments. Millimetre basket anchors and float32 staged placement preserve
the geometric preparation recipe; they are not changes to physics tolerances.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tablecloth_rule_sim import Simulation, folded_initial

def basket_supported_fold(uv,size,bounds,support_top):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Finite measured basket bounds are required")
    if not np.isfinite(support_top) or support_top >= bounds[1, 2] or support_top <= bounds[0, 2] + .026:
        raise ValueError("Visible linen support must lie inside the basket")
    # Measured House basket anchors are specified to the nearest millimetre.
    wall=round(float(bounds[0,1])+.028,3)
    rim=round(float(bounds[1,2]),3)
    floor=round(float(bounds[0,2])+.029,3)
    q=folded_initial(uv,size,np.zeros(3),0.,2)
    q-=q.mean(0)
    q=Rotation.from_euler('x',-75,degrees=True).apply(q)
    q[:,0]+=round(float(bounds[:,0].mean()),3)
    q[:,2]+=floor-q[:,2].min()
    below = q[:,2] < rim
    if not below.any():
        raise ValueError("Initial cloth must extend below the basket rim")
    q[:,1]+=wall-q[below,1].min()
    q=q.astype(np.float32)
    q[:,2]+=max(0.,support_top+.004-floor)
    below = q[:,2] < rim
    if not below.any():
        raise ValueError("Initial cloth must extend below the basket rim")
    q[:,1]+=wall-q[below,1].min()
    return q

class SupportedCanvasInitial(Simulation):
    def __init__(self, args):
        if args.initial or args.replay_action or args.start_held or args.mode != "spread":
            raise ValueError("Prepare free canvas independently, without a previous episode")
        super().__init__(args)
        q = basket_supported_fold(self.uv, self.size, self.basket_bounds, args.linen_support_top)
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite geometric initial cloth")
        for state in (self.a, self.b):
            state.particle_q.assign(q)
            state.particle_qd.zero_()
        self.initial = q.copy()
        self.events.append(dict(time=0., event="unpinned_supported_basket_initialization",
                                lean_from_vertical_deg=15., basket_anchor_precision_m=.001,
                                support_surface_clearance_m=.004, cloth_fixed_vertices=0))

    def run_policy(self):
        targets = self.hands.centres()
        for _ in range(max(1, round(self.args.settle * self.args.fps))):
            self.advance("initial_drape_settle", targets, {"L": 0., "R": 0.}, targets, False)

    def save(self, failure=None):
        super().save(failure)
        path = self.output / "trial.json"
        metadata = json.loads(path.read_text())
        metadata.update(trial_role="scene_initialization_not_task_data",
                        initial_state_model="free folded canvas on measured basket and visible packed linens",
                        owner_initialization_entrypoint=str(Path(__file__).resolve()),
                        actual_settling_seconds=len(self.frames) / self.args.fps,
                        completed_benchmark_trajectory=False)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
