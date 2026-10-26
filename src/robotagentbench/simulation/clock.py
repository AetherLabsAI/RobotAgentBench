"""Measured native scene clock; no autonomous physics stepping."""

class NativeSceneClock:
    """Port for the existing explicit sim.simulate(dt,t)/fetch_results runner.

snapshot must return immutable bytes of ALL measured native poses/velocities;
read_time returns the runner's accumulated simulation time, never steps/hz.
This class does not step, reset, save, move bodies or alter renderer settings.
"""
    def __init__(self, stage, simulation, read_time, snapshot):
        self.stage, self.simulation = stage, simulation
        self.read_time, self.snapshot = read_time, snapshot
        self.attribute = stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physxScene:timeStepsPerSecond')
        if not self.attribute or self.hz not in (60, 240):
            raise ValueError('Expected existing 60/240 Hz scene')

    @property
    def hz(self): return self.attribute.Get()

    @property
    def current_time(self): return float(self.read_time())

    def get_physics_dt(self): return 1/self.hz

    def set_hz(self, hz):
        if hz not in (60, 240): raise ValueError('Unsupported clock')
        from pxr import Usd
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            self.attribute.Set(hz)
        self.simulation.flush_changes()
