"""Native reasoning ports: existing contact collector and selection lifecycle."""
from pathlib import Path
import json
from short_task_ContextReasoning.tasks.select_trash.task import NativeEvidence
from short_task_ContextReasoning.tasks.select_trash.native import SelectionPort, SelectionSession
from .task import ShelfSelectionEvaluator
from short_task_ContextReasoning.tasks.select_sofa.task import SofaSelectionEvaluator

class Evidence(NativeEvidence):
    evaluator_class=ShelfSelectionEvaluator
    def __init__(self,stage,plan,episode_id):
        from openend.tasks.meal_tray.task_contact_routing import NativeContactRoutes
        self.bindings={a['instance_id']:a['path'] for a in plan['candidates']}
        self.native=NativeContactRoutes(stage,{k:(v,) for k,v in self.bindings.items()})
        self.evaluator=self.evaluator_class(episode_id,plan)
        self.episode_id=episode_id

class SofaEvidence(Evidence):
    evaluator_class=SofaSelectionEvaluator

class ContextSelectionPort(SelectionPort):
    def accept_initial(self):
        # Preserve measured evidence for failed admissions; do not weaken checks.
        (self.output/'initial_contact_diagnostic.json').write_text(json.dumps(dict(
            support_root=self.support_root,bodies=self.bodies,history=list(self.history),
            current_pairs=[dict(paths=list(paths),samples=list(samples))
                           for paths,samples in self.contacts.pairs()]),default=str,indent=2)+'\n')
        return super().accept_initial()

def build_session(request,stage,manifest,output,*,episode_id,workspace_selector=None,
                  sensor=None,runtime_settings=None):
    from short_tasks import registry
    from short_tasks.scene import verify_scene_sources, verify_manifest
    plan=registry.bind(registry.resolve(request),manifest)
    verify_scene_sources(); verify_manifest(manifest)
    plan['source_receipt']=registry.verify_sources(plan)
    sofa=plan['task_name']=='select_sofa'
    if sofa:
        support='/World/Assets/Sofa'
    else:
        support='/World/RoomEpisode/Shelf'
    if not stage.GetPrimAtPath(support): raise ValueError('Actual selection support is missing')
    if workspace_selector is None:
        if sensor is None: raise ValueError('Existing calibrated sensor required')
        from openend.tasks.all_objects_integration.workspace_physics import from_scene
        workspace_selector=from_scene(stage,manifest,sensor)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    name=plan['task_name']
    port=ContextSelectionPort(stage,plan['selection'],output/name,episode_id,
        evidence_factory=SofaEvidence if sofa else Evidence,support_root=support,
        support_roots=tuple(p for p in (support,'/World/Assets/SeatOverlay') if stage.GetPrimAtPath(p))
                      if sofa else (support,))
    return SelectionSession(plan,manifest,{name:port},episode_id=episode_id,workspace_selector=workspace_selector)
