"""Per-worker scene-wide clock. No reset, pose edits, or global module mutation."""
import math

SCHEMA = 'arena.physics_rate.v1'
SUPPORTED_HZ = (60, 120, 240)
DEFAULT_HZ = 240
CONTROL_HZ = 20


def validate_hz(hz):
    if type(hz) is not int or hz not in SUPPORTED_HZ:
        raise ValueError('physics_hz must be an integer: 60, 120 or 240')
    return hz


def evidence_substeps(execution, expected_hz=DEFAULT_HZ):
    """Legacy receipts mean 240 Hz; new receipts must match the issued schedule."""
    validate_hz(expected_hz)
    receipt = execution.get('physics_rate')
    if receipt is None:
        if expected_hz != DEFAULT_HZ:
            raise ValueError('Missing switched physics clock evidence')
        return DEFAULT_HZ // CONTROL_HZ
    if (receipt.get('schema') != SCHEMA or receipt.get('physics_hz') != expected_hz
            or receipt.get('substeps') != expected_hz // CONTROL_HZ
            or not math.isclose(receipt.get('physics_dt_s', -1), 1 / expected_hz, rel_tol=0, abs_tol=1e-10)
            or receipt.get('control_dt_s') != 1 / CONTROL_HZ):
        raise ValueError('Mismatched physics clock evidence')
    return expected_hz // CONTROL_HZ


class PhysicsRate:
    def __init__(self):
        self.hz = DEFAULT_HZ
        self.revision = 0

    @property
    def dt(self):
        return 1 / self.hz

    @property
    def substeps(self):
        return self.hz // CONTROL_HZ

    def receipt(self):
        return dict(schema=SCHEMA, physics_hz=self.hz, physics_dt_s=self.dt,
                    control_dt_s=1 / CONTROL_HZ, substeps=self.substeps,
                    revision=self.revision, scope='entire_physics_scene')

    def switch(self, world, hz, *, setter, helper):
        """Caller serializes at a control boundary and fails closed on any error.

        setter authors only the session-layer physics clock. Never render,
        update the app, reset, stop/play, or take a hidden physics step here.
        """
        validate_hz(hz)
        if not math.isclose(world.get_physics_dt(), self.dt, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError('Unexpected physics clock before switch')
        helper.validate_physics_dt(1 / hz)
        before = (float(world.current_time), int(world.current_time_step_index))
        previous = self.receipt()
        if hz != self.hz:
            setter(1 / hz)
            if not math.isclose(world.get_physics_dt(), 1 / hz, rel_tol=0, abs_tol=1e-10):
                raise RuntimeError('Native physics clock switch did not take effect')
            if before != (float(world.current_time), int(world.current_time_step_index)):
                raise RuntimeError('Physics clock switch advanced or reset simulation')
            helper.set_physics_dt(1 / hz)
            self.hz = hz
            self.revision += 1
        return dict(previous=previous, current=self.receipt(), time_s=before[0],
                    physics_step_index=before[1], changed=previous['physics_hz'] != hz)
