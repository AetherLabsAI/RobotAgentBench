"""Change real USD light intensity in an anonymous episode layer."""
import json
import math
from pathlib import Path

from .cycle import LightingCycle

HERE = Path(__file__).resolve().parent


def light_records(stage):
    from pxr import UsdLux, UsdGeom
    rows = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            continue
        light = UsdLux.LightAPI(prim)
        attr = light.GetIntensityAttr()
        value = attr.Get()
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('Invalid scene light intensity: ' + str(prim.GetPath()))
        if attr.GetNumTimeSamples() or attr.HasAuthoredConnections():
            raise ValueError('Existing animated/connected light requires explicit adaptation: ' + str(prim.GetPath()))
        if float(value) == 0:
            continue
        rows.append(dict(path=str(prim.GetPath()), attribute=str(attr.GetPath()),
                         baseline_intensity=float(value), type=prim.GetTypeName()))
    if not rows:
        raise ValueError('No active positive-intensity USD scene lights; do not fake image brightness')
    return sorted(rows, key=lambda row: row['path'])



def permitted_audit_output(output):
    output = Path(output).resolve()
    batches = [Path('/mnt/agent-bench-data/lingjun/astra_all14_20261015/codex_robustness6_v1/stock_wooden_tray_lighting_a3/native'),
               Path('/mnt/agent-bench-data/lingjun/astra_all14_20261015/codex_robustness120_v1/stock_wooden_tray_lighting/native')]
    return output.is_relative_to(HERE) or any(
        batch.resolve() == batch and output.is_relative_to(batch)
        and output.name == 'lighting.json' for batch in batches)


class SceneLighting:
    def __init__(self, stage, seed, output=None):
        if not stage.GetSessionLayer().anonymous:
            raise ValueError('Independent anonymous episode layer required')
        self.stage = stage
        self.records = light_records(stage)
        self.cycle = LightingCycle(seed)
        self.closed = False
        self.events = []
        self.output = None if output is None else Path(output).resolve()
        if self.output is not None and not permitted_audit_output(self.output):
            raise ValueError('Lighting audit must stay inside this task')
        self.prior_defaults = {}
        layer = stage.GetSessionLayer()
        for row in self.records:
            spec = layer.GetAttributeAtPath(row['attribute'])
            self.prior_defaults[row['attribute']] = (spec.default if spec and spec.HasInfo('default') else None)

    def arm(self, now):
        self.cycle.arm(now)
        self._save()

    def tick(self, now):
        if self.closed:
            raise RuntimeError('Closed lighting controller')
        changes = self.cycle.tick(now)
        if changes:
            from pxr import Usd
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                for row in self.records:
                    attr = self.stage.GetAttributeAtPath(row['attribute'])
                    attr.Set(row['baseline_intensity'] * self.cycle.multiplier)
            self.events.extend(changes)
            self._save()
        return changes

    def snapshot(self):
        return dict(schema='arena.shelf_lighting_cycle.v1', seed=self.cycle.seed,
                    started=self.cycle.started, state='bright' if self.cycle.bright else 'dark',
                    multiplier=self.cycle.multiplier, interval_s=[2., 3.],
                    lights=self.records, events=self.events, closed=self.closed,
                    physics_parameter_writes=0, object_pose_writes=0,
                    camera_parameter_writes=0, image_postprocessing=False)

    def _save(self):
        if self.output is not None:
            self.output.write_text(json.dumps(self.snapshot(), indent=2) + '\n')

    def close(self):
        if self.closed:
            return
        from pxr import Usd
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for row in self.records:
                attr = self.stage.GetAttributeAtPath(row['attribute'])
                previous = self.prior_defaults[row['attribute']]
                if previous is None:
                    attr.Clear()
                else:
                    attr.Set(previous)
        self.closed = True
        self._save()
