"""Prepare unpinned canvas against the basket wall, without reference episodes."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tablecloth_rule_sim import Simulation, folded_initial, main


def basket_wall_fold(uv, size, basket_bounds, support_top):
    """Place a broad folded sheet 15 degrees from vertical over packed linens."""
    q = folded_initial(uv, size, np.zeros(3), 0., 2)
    q -= q.mean(0)
    q = Rotation.from_euler('x', -75, degrees=True).apply(q)
    q[:, 0] += basket_bounds[:, 0].mean()
    floor = max(float(basket_bounds[0, 2]) + .024, support_top)
    q[:, 2] += floor + .004 - q[:, 2].min()
    below_rim = q[:, 2] < float(basket_bounds[1, 2])
    if not below_rim.any():
        raise ValueError('Packed-linen support leaves no cloth below the basket rim')
    q[:, 1] += float(basket_bounds[0, 1]) + .028 - q[below_rim, 1].min()
    return q


def basket_single_fold(uv, size, basket_bounds):
    """One material fold, leaning on the visible basket; all vertices are free.

    Geometry is an episode initializer. Gravity, basket contact and self-contact
    determine its resting shape before and throughout the removal controller.
    """
    u, v = uv.T
    angle = np.deg2rad(15.)
    x = np.minimum(v, size[1] - v) - size[1] / 4
    floor = float(basket_bounds[0, 2]) + .024
    return np.column_stack((basket_bounds[:, 0].mean() + x,
        basket_bounds[0, 1] + .082 - u * np.sin(angle) + .022 * (v - size[1] / 2),
        floor + .005 + u * np.cos(angle)))


class CanvasInitial(Simulation):
    def __init__(self, args):
        if args.initial or args.replay_action or args.start_held or args.mode != 'spread':
            raise ValueError('Canvas preparation requires spread mode and no prior cloth state')
        super().__init__(args)
        q = basket_wall_fold(self.uv, self.size, self.basket_bounds, args.linen_support_top)
        for state in (self.a, self.b):
            state.particle_q.assign(q)
            state.particle_qd.zero_()
        self.initial = q.copy()
        self.events.append(dict(time=0., event='unpinned_basket_wall_initialization',
                                lean_from_vertical_deg=15., wall_inset_m=.028,
                                support_surface_clearance_m=.004, cloth_fixed_vertices=0))

    def run_policy(self):
        targets = self.hands.centres()
        for _ in range(max(1, round(self.args.settle * self.args.fps))):
            self.advance('initial_drape_settle', targets, {'L': 0., 'R': 0.}, targets, False)

    def save(self, failure=None):
        super().save(failure)
        path = self.output / 'trial.json'
        metadata = json.loads(path.read_text())
        metadata.update(trial_role='scene_initialization_not_task_data',
                        initial_state_model='basket-wall folded canvas supported by basket and packed linens',
                        owner_initialization_entrypoint=str(Path(__file__).resolve()),
                        completed_benchmark_trajectory=False)
        path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main(simulation_class=CanvasInitial)
