"""Episode-wide native soft-body capability, independent of task scoring.

GPU dynamics is not the tensor backend: keep existing numpy/CPU articulation
interfaces, clocks and solver settings. No automatic attachment is installed.
All USD changes are session-only and occur before the first physics reset.
"""
import math

from .physics_defaults import PHYSICS_DT

PROFILE = 'arena.soft_body.gpu_surface.v1'
BETA_KEY = '/persistent/physics/enableDeformableBeta'


class SoftBodyRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.previous_beta = settings.get(BETA_KEY)
        self.stage = None
        self.closed = False
        self.gpu_dynamics = True
        self.broadphase = 'GPU'
        settings.set(BETA_KEY, True)

    def prepare(self, stage, world):
        from pxr import PhysxSchema, Usd
        if self.closed or self.stage is not None:
            raise RuntimeError('Soft-body runtime prepares once per episode')
        if not math.isclose(world.get_physics_dt(), PHYSICS_DT, rel_tol=0, abs_tol=1e-10):
            raise RuntimeError('Soft-body support must not change the public physics clock')
        layer = stage.GetEditTarget().GetLayer()
        if layer != stage.GetSessionLayer():
            raise RuntimeError('Soft-body capability requires the session edit layer')
        scene = stage.GetPrimAtPath('/World/PhysicsScene')
        if not scene:
            raise RuntimeError('Expected existing physics scene')
        # World(device=cpu) resets these. Set once AFTER World construction.
        # Do not alter solver type, CCD, buffers, dt, robot, assets or materials.
        with Usd.EditContext(stage, layer):
            api = PhysxSchema.PhysxSceneAPI.Apply(scene)
            api.CreateEnableGPUDynamicsAttr(self.gpu_dynamics)
            api.CreateBroadphaseTypeAttr(self.broadphase)
        self.stage = stage
        return self.verify(world)

    def verify(self, world, *, expected_dt=PHYSICS_DT):
        if self.closed or self.stage is None:
            raise RuntimeError('Soft-body runtime is not prepared')
        scene = self.stage.GetPrimAtPath('/World/PhysicsScene')
        if (self.settings.get(BETA_KEY) is not True
                or scene.GetAttribute('physxScene:enableGPUDynamics').Get() is not self.gpu_dynamics
                or scene.GetAttribute('physxScene:broadphaseType').Get() != self.broadphase
                or (self.gpu_dynamics and expected_dt != PHYSICS_DT)
                or not math.isclose(world.get_physics_dt(), expected_dt, rel_tol=0, abs_tol=1e-10)):
            raise RuntimeError('Episode-wide soft-body capability changed')
        return dict(profile=PROFILE if self.gpu_dynamics else 'arena.rigid.cpu_mbp.v1',
                    gpu_dynamics=self.gpu_dynamics, broadphase=self.broadphase,
                    deformable_beta=True, physics_dt_s=expected_dt,
                    lifetime='entire_episode', automatic_grasp_assistance=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.previous_beta is None:
            self.settings.destroy_item(BETA_KEY)
        else:
            self.settings.set(BETA_KEY, self.previous_beta)


def gpu_requirement_reasons(stage, task=None):
    """Trusted scene/task evidence only; never a network-selected physics mode.

    A task creating soft objects later must declare requires_gpu_dynamics=True.
    Cloth-transition hooks and predeclared cloth markers are conservative GPU
    requirements even before a deformable is activated. False cannot override
    an actual scene requirement. Selection is fixed for the entire episode.
    """
    from .task_runtime import WorkerTask
    reasons = []
    if task is not None:
        required = getattr(task, 'requires_gpu_dynamics', False)
        if type(required) is not bool:
            raise ValueError('requires_gpu_dynamics must be a trusted boolean')
        if required:
            reasons.append('task.requires_gpu_dynamics')
        hook = getattr(task, 'scene_transitions', None)
        function = getattr(hook, '__func__', hook)
        if function is not None and function is not WorkerTask.scene_transitions:
            reasons.append('task.cloth_transition_hook')
    for prim in stage.Traverse():
        schemas = [*prim.GetAppliedSchemas(), prim.GetTypeName()]
        if any(any(word in schema.lower() for word in ('deformable', 'particle', 'softbody'))
               for schema in schemas):
            reasons.append('scene.gpu_body:' + str(prim.GetPath()))
        if prim.GetCustomDataByKey('arena:transitionRole') == 'cloth':
            reasons.append('scene.cloth_transition:' + str(prim.GetPath()))
    return tuple(sorted(set(reasons)))


class EpisodePhysicsRuntime(SoftBodyRuntime):
    """Select once before reset; no CPU fallback after GPU errors.

    GPU soft-body behavior is unchanged. Rigid scenes retain the original
    CPU/MBP physics path. Tensor backend, GPU cameras, clock, solvers, collisions
    and materials are independent and are not changed by this selection.
    """
    def prepare(self, stage, world, *, task=None):
        if self.closed or self.stage is not None:
            raise RuntimeError('Episode physics prepares once per episode')
        self.task = task
        self.selection_reasons = gpu_requirement_reasons(stage, task)
        self.gpu_dynamics = bool(self.selection_reasons)
        self.broadphase = 'GPU' if self.gpu_dynamics else 'MBP'
        return super().prepare(stage, world)

    def verify(self, world, *, expected_dt=PHYSICS_DT):
        receipt = super().verify(world, expected_dt=expected_dt)
        if not self.gpu_dynamics and gpu_requirement_reasons(self.stage, self.task):
            raise RuntimeError('A GPU-only body/transition was introduced into a rigid CPU episode')
        return dict(receipt, selection='trusted_scene_and_task.v1',
                    selection_reasons=list(self.selection_reasons))
