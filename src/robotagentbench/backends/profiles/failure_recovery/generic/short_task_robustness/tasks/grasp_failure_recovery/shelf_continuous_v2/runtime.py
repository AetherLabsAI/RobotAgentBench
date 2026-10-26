"""Original per-product physical reader and unchanged recoverable pick scorer."""
import json
from pathlib import Path
from ..common import CURRENT,owned,require_source,verify_sources

def make_runtime(context,initial_file):
    from evaluation.tasks.shelf_pick import api_runtime as source,initial_state
    from short_tasks.tasks.stock_wooden_tray.adapter import RecoverableShelfProcess
    from .catalog import APIS,POLICY
    require_source(source,CURRENT);initial_file=owned(initial_file);initial=json.loads(initial_file.read_text())
    if initial['source_api'] not in APIS or context.task_id!=initial['source_api'] or initial['initialization_policy']!=POLICY:raise ValueError('Source product/runtime mismatch')
    verify_sources(initial['source_layers']);verify_sources(initial['code_sources'])
    class PreparedShelf(source.ShelfPickRuntime):
        def __init__(self):
            super().__init__(context)
            self.scorer=RecoverableShelfProcess(context.episode_id,source.PHYSICS_DT,context.task_id)
            self.snapshot=initial_file.parent/'source_snapshot.json';snapshot=json.loads(self.snapshot.read_text())
            self.plan=dict(initial,api_id=context.task_id,targets=[initial['target']],
                robot_initial_positions=initial['robot_initialization']['positions'],stock_roots=list(initial_state.STOCK_ROOTS),
                dependency_sha256=initial['source_layers'],source_episode=dict(snapshot['source'],
                    item_supports={item['prim_path']:item['support'] for item in snapshot['items']}))
        def initialize(self,source):return Path(self.plan['scene'])
        def randomize(self,scene):return scene
    return PreparedShelf()

