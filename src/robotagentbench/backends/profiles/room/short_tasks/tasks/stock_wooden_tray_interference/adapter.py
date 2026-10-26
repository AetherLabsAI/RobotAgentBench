"""Independent registration; original shelf adapter remains unchanged."""
from copy import deepcopy

from short_tasks.tasks.stock_wooden_tray import adapter as base
from short_tasks.tasks.stock_wooden_tray._contract import options, targets, summaries, number, result
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
from .unique_target import require_unique_target
from .contact_award import VERSION, TargetContactAward, observe_stream, apply_generic

TASK_NAME = 'stock_wooden_tray_interference'
RULE_VERSION = 'openend.short_task.stock_wooden_tray_interference.v1'
ACTOR_CONTRACT = 'shelf_human_relocation.v1'


def resolve(request):
    selected = options(request, TASK_NAME, allowed=('relocation_distance',))
    from .distance_bands import resolve_selection
    distance = resolve_selection(selected.get('relocation_distance'))
    drinks = {key: value for key, value in PRODUCTS.items() if value.tier == 'tier3'}
    products=targets(request, drinks, maximum=1)
    if request.get('destination') is not None:
        raise ValueError('This interference task ends after pickup; no destination is requested')
    if any(extra.get('asset')==products[0] for extra in request.get('extras',[]) if isinstance(extra,dict)):
        raise ValueError('An extra object cannot duplicate the target product')
    source = deepcopy(request)
    source['options'] = {}  # This option belongs only to the interference task.
    source.update(task_name=base.TASK_NAME, rule_version=base.RULE_VERSION)
    plan = base.resolve(source)
    plan.update(
        task_name=TASK_NAME, rule_version=RULE_VERSION,
        runtime_adapter=__name__,
        scene_preparation='short_tasks.tasks.stock_wooden_tray_interference.prepare_scene.prepare_reserved',
        environment_actor_contract=ACTOR_CONTRACT,
        human_physics_backend='cpu_physx',
        relocation_distance=distance,
        instruction='After the human hand relocates it, find and pick up the only '+base.product_label_en(products[0])+' on the shelf. Placing it on a tray is not required.',
        score_policy=dict(maximum=5,process_maximum=5,final_maximum=0,arithmetic='component_defined'),
        scoring_revision=VERSION,
        completion_policy='original_pick_success_after_human_intervention',
        initialization_constraints=dict(unique_product=products[0],whole_room_count=1,
            background_excluded_products=products),
        interference=dict(duration_s=3.0, count=1, same_instance=True,
                          same_shelf_level=True, hand_and_forearm_collision=True),
        available=True,
        native_validation='trigger25_public_sessions_layout03_46_v1',
        validation_scope='25 cm trigger and complete human relocation verified in layout03/seed8 and layout46/seed9 with continuous recorded robot actions, native shelf support and zero human/robot collision episodes in these samples. Every episode requires admission; broader random coverage and model recovery/pickup remain unverified.',
    )
    plan['initial_requirements'] = [
        'Exactly one instance of the target product in the whole room, including background objects and other sources.',
        'The target starts supported by its shelf, unheld. No wooden-tray placement goal.',
        'Human hand is absent before proximity activation, collidable during interference, and absent after retreat.',
        'Relocate only before robot contact; keep the exact target on the same shelf level.',
        'Start once only when either native jaw center is within 0.25 m of the target bounding box; never start on a timer.',
        'External manipulation must never earn robot pickup credit.',
        'During initialization, reserve one full target-sized empty slot on the same shelf level, separate from scoring objects.',
        'Admit and freeze the approach/transport/exit route before activation; background stock must not occupy the reserved slot.',
        'Use a clear accessible target position; do not solve a blocked rear-row reach at runtime.',
        'Before activation recheck the reserved slot; if occupied, do not start or search for another destination.',
    ]
    return plan


def bind(plan, manifest):
    from .prepare_scene import filtered_manifest
    from .distance_bands import require_manifest_band
    require_manifest_band(plan, manifest)
    # Binding previews the task-local scene filter. The scene hook applies it
    # before PhysX; the native port still independently requires uniqueness.
    manifest, _ = filtered_manifest(plan, manifest)
    product=plan['target_categories'][0]
    unique=require_unique_target(manifest,product)
    # Base binding still supplies the existing native reader; score semantics
    # and public completion are replaced only in this independent task.
    if unique.get('task_id')!=plan['runtime_components'][0]['instance_id']:
        raise ValueError('The unique product must belong to this task')
    return base.bind(plan, manifest)


def normalize(plan, component_summaries):
    values=summaries(plan,component_summaries);rows=[]
    for declaration in plan['runtime_components']:
        identity=declaration['instance_id'];s=values[identity]
        if (s.get('task_kind')!='shelf_pick_interference' or s.get('maximum_score')!=4
                or s.get('task_parameters')!=declaration['parameters']):
            raise ValueError('Expected exact pickup-only interference component')
        pick=s.get('pick') or {}
        process=number(pick.get('earned_points',0))
        deductions=sum(number(pick.get(k,0)) for k in ('collision_deduction','drop_back_deduction','wrong_grasp_deduction','hold_deduction'))
        complete=bool(pick.get('success',False) and s.get('operation_complete',False) and s.get('success',False))
        rows.append(dict(instance_id=identity,category=declaration['parameters']['product'],
            process=process,current_final=0,deductions=deductions,process_complete=complete,success=complete,
            component_total=number(pick.get('score',0)),phase=s.get('phase'),committed_target=s.get('committed_target')))
    # EnvironmentActor separately requires exercised interference and applies
    # the existing human/robot collision charges. Human motion earns no points.
    return apply_generic(plan,rows,values,result)


def create_environment_actor(stage, manifest, output, episode_id):
    from .cpu_contract import validate_cpu_scene
    validate_cpu_scene(stage)
    from .environment import EnvironmentActor
    from .native_backend import NativeBackend
    return EnvironmentActor(NativeBackend(stage, manifest, output, episode_id))


def factory_override(plan, declaration):
    Parent = base.factory_override(plan, declaration)

    class InterferencePort(Parent):
        """Suppress scripted pickup without changing any original score rule."""
        external_control = False
        pickup_unlocked = False

        def __init__(self,stage,manifest,output,episode_id,instance_id,parameters):
            require_unique_target(manifest,parameters['product'])
            super().__init__(stage,manifest,output,episode_id,instance_id,parameters)
            from .pick_only import PickOnlyComposite
            self.composite=PickOnlyComposite(self.scene.identity,room_managed=True)
            self.score=self.composite.summary()
            self.target_contact=TargetContactAward(episode_id,'shelf_move',self.scene.identity.candidates[0])
            self.contact_sample=0
            self.contact_armed=False

        def accept_initial(self):
            receipt=super().accept_initial()
            self.contact_armed=True
            return receipt

        def _record_target_contact(self):
            observe_stream(self.target_contact,self.source.contacts,sample_index=self.contact_sample,
                time_s=self.clock.current_time,admitted=self.contact_armed,
                relocation_complete=self.pickup_unlocked and not self.external_control)
            self.contact_sample+=1

        def _contact_status(self):
            super()._contact_status()
            self.score['target_contact']=self.target_contact.snapshot()

        @property
        def workspace_id(self):
            return 'shelf'

        def set_external_control(self, enabled):
            if type(enabled) is not bool:
                raise ValueError('Trusted external-control boolean required')
            if enabled == self.external_control:
                return
            if enabled:
                # Trigger must precede robot pickup, never steal earned progress.
                if self.composite.summary()['pick'].get('earned_points', 0):
                    raise RuntimeError('Cannot intervene after robot pickup progress')
                self.buffer.clear()
                self.composite.pause()
            self.source.detector.reset()
            self.source.locked = False
            if not enabled:
                raw = self.source.latest
                if raw is None or not raw['contacts']['target_supported']:
                    raise RuntimeError('Relocated target lacks measured shelf support')
                self.source.initial_z = {
                    p: float(raw['poses'][p]['position_m'][2])
                    for p in self.scene.identity.candidates}
            self.external_control = enabled

        def unlock_pickup_after_intervention(self, audit):
            from .intervention_rules import InterventionAudit, Phase
            if not isinstance(audit, InterventionAudit) or audit.phase != Phase.DONE:
                raise RuntimeError('Measured completed relocation audit required')
            if self.pickup_unlocked or self.external_control:
                raise RuntimeError('Pickup unlock requires a completed, released intervention')
            if tuple(self.scene.identity.candidates) != (audit.target,):
                raise ValueError('Relocation and pickup must use the same unique target')
            raw = self.source.latest
            if raw is None or not raw['contacts']['target_supported']:
                raise RuntimeError('Relocated target lacks current native shelf support')
            if self.composite.summary()['pick'].get('earned_points',0):
                raise RuntimeError('Pickup credit existed before relocation completion')
            self.composite.pause()
            self._short_started = False
            self.pickup_unlocked = True

        def after_physics(self, report):
            if self.pickup_unlocked and not self.external_control:
                value=super().after_physics(report)
            else:
                self.source.after_physics(report)
                self.source.detector.reset()
                self.source.locked = False
                # Deliberately no scoring frames for externally carried target.
                self.buffer.clear()
                value=None
            self._record_target_contact()
            return value

        def consume_control(self, active):
            if self.pickup_unlocked and not self.external_control:
                return super().consume_control(active)
            self.discard_control()
            self.score = self.composite.summary()
            self.score['target_contact']=self.target_contact.snapshot()
            return self.score

    return InterferencePort
