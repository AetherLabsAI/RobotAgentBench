"""Retain ordinary shelf scoring; add only task-local scene illumination."""
from copy import deepcopy
from pathlib import Path

from short_tasks.tasks.stock_wooden_tray import adapter as base
from short_tasks.tasks.stock_wooden_tray._contract import options, summaries, number, result
from evaluation.tasks.shelf_pick.openended_composite_v1.catalog import PRODUCTS
from .contact_award import VERSION, TargetContactAward, observe_stream, apply_generic

TASK_NAME = 'stock_wooden_tray_lighting'
RULE_VERSION = 'openend.short_task.stock_wooden_tray_lighting.v1'


def resolve(request):
    options(request, TASK_NAME)
    if request.get('destination') is not None:
        raise ValueError('Lighting task ends after pickup; no placement destination')
    source = deepcopy(request)
    source.update(task_name=base.TASK_NAME, rule_version=base.RULE_VERSION)
    plan = base.resolve(source)
    plan.update(task_name=TASK_NAME, rule_version=RULE_VERSION, runtime_adapter=__name__,
                scene_preparation='short_tasks.tasks.stock_wooden_tray_lighting.adapter.prepare_scene',
                lighting=dict(bright_multiplier=1., dark_multiplier=.25,
                              interval_s=[2., 3.], transition='instant', start='after_initial_admission',
                              seed=plan['scene_request']['seed']),
                instruction='Pick up the designated shelf products: '+ ', '.join(base.product_label_en(p) for p in plan['target_categories'])+'. Placing them on a tray is not required.',
                score_policy=dict(maximum=4*len(plan['target_categories'])+1,
                                  process_maximum=4*len(plan['target_categories'])+1,
                                  final_maximum=0, arithmetic='component_defined'),
                scoring_revision=VERSION,
                completion_policy='original_pick_success_only',
                available=True, native_validation='native_lighting_preview_002',
                validation_scope='Real light switching rendered; pickup scoring tested on typed facts. Scored physical admission and complete actions remain unverified.')
    plan['initial_requirements'] = list(plan['initial_requirements']) + [
        'Existing scene lights must be measurable and static before the lighting cycle.',
        'Keep original light intensity during initialization; switch bright/dark only after task admission.',
        'Preserve ordinary target, background, robot station and posture distributions.',
    ]
    return plan


def bind(plan, manifest):
    return base.bind(plan, manifest)


def normalize(plan, component_summaries):
    values = summaries(plan, component_summaries); rows = []
    for declaration in plan['runtime_components']:
        identity = declaration['instance_id']; s = values[identity]
        if (s.get('task_kind') != 'shelf_pick_lighting' or s.get('maximum_score') != 4
                or s.get('task_parameters') != declaration['parameters']):
            raise ValueError('Expected exact pickup-only lighting component')
        pick = s.get('pick') or {}
        complete = bool(pick.get('success',False) and s.get('operation_complete',False) and s.get('success',False))
        rows.append(dict(instance_id=identity, category=declaration['parameters']['product'],
            process=number(pick.get('earned_points',0)), current_final=0,
            deductions=sum(number(pick.get(k,0)) for k in
                ('collision_deduction','drop_back_deduction','wrong_grasp_deduction','hold_deduction')),
            process_complete=complete, success=complete, component_total=number(pick.get('score',0)),
            phase=s.get('phase'), committed_target=s.get('committed_target')))
    out = apply_generic(plan, rows, values, result)
    evidence = {identity: deepcopy(s['contact_evidence']) for identity,s in values.items()
                if 'contact_evidence' in s}
    out['contact_evidence'] = evidence
    out['evidence_coverage_complete'] = all(v.get('gap_steps',0)==0 for v in evidence.values())
    out['scoring_status'] = ('waiting_for_contact_evidence' if any(not v['complete'] for v in evidence.values()) else 'current')
    return out


def prepare_scene(plan, stage, manifest):
    from .lighting import light_records
    result = deepcopy(manifest)
    result['lighting_variant'] = dict(task_name=TASK_NAME, profile=deepcopy(plan['lighting']),
                                      initial_lights=light_records(stage),
                                      initial_intensity_writes=0)
    return result


def factory_override(plan, declaration):
    """Use the existing native-port extension, without an extra physics actor."""
    OriginalPort = base.factory_override(plan, declaration)
    owner = declaration['instance_id'] == plan['runtime_components'][0]['instance_id']
    from .lighting import SceneLighting

    class LightingStockPort(OriginalPort):
        def __init__(self, stage, manifest, output, *args, **kwargs):
            super().__init__(stage, manifest, output, *args, **kwargs)
            self.target_contact=TargetContactAward(args[0] if args else kwargs['episode_id'],
                'shelf_light',self.scene.identity.candidates[0])
            self.contact_sample=0
            self.contact_armed=False
            self.light_cycle = None
            try:
                from .pick_only import PickOnlyComposite
                self.composite = PickOnlyComposite(self.scene.identity, room_managed=True)
                self.score = self.composite.summary()
                if owner:
                    self.light_cycle = SceneLighting(stage, plan['lighting']['seed'], Path(output) / 'lighting.json')
            except Exception:
                super().close()
                raise

        @property
        def workspace_id(self):
            return 'shelf'

        def activation_geometry(self):
            if self.source is not None and getattr(self.source,'latest',None) is not None:
                from evaluation.tasks.shelf_pick.geometry import rotation
                bounds = []
                for path in self.scene.identity.candidates:
                    pose = self.source.latest['poses'][path]
                    vertices = rotation(pose['quaternion_wxyz']).apply(self.scene.workspace.collision_vertices[path])+pose['position_m']
                    bounds.append([vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()])
                return bounds
            targets = [o for o in self.manifest['objects'] if o['path']+'/Asset' in self.scene.identity.candidates]
            if not targets: raise ValueError('Missing pickup activation bounds')
            return [o['visible_bounds_world_m'] for o in targets]

        def consume_control(self, active):
            if type(active) is not bool:
                raise ValueError('Measured activation boolean required')
            if self.ended:
                # Multi-target episodes may continue after one pickup completes.
                self.discard_control()
                self.score = self.composite.summary()
                self._contact_status()
                return self.score
            return super().consume_control(active)

        def accept_initial(self):
            receipt = super().accept_initial()
            self.contact_armed=True
            if self.light_cycle is not None:
                self.light_cycle.arm(float(self.clock.current_time))
            return receipt

        def after_physics(self, report):
            super().after_physics(report)
            observe_stream(self.target_contact,self.source.contacts,sample_index=self.contact_sample,
                time_s=self.clock.current_time,admitted=self.contact_armed,relocation_complete=True)
            self.contact_sample+=1
            if self.light_cycle is not None:
                self.light_cycle.tick(float(self.clock.current_time))

        def _contact_status(self):
            super()._contact_status()
            self.score['target_contact']=self.target_contact.snapshot()

        def after_inactive(self, report):
            super().after_inactive(report)
            if self.light_cycle is not None:
                self.light_cycle.tick(float(self.clock.current_time))

        def close(self):
            try:
                if self.light_cycle is not None:
                    self.light_cycle.close()
            finally:
                super().close()

    return LightingStockPort
