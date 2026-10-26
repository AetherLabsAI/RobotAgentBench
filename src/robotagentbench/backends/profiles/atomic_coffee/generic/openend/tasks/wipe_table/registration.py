"""Exact generated brush/stain binding. No physics or scene mutation."""
from pathlib import Path
import hashlib

KIND = 'wipe_table'
INSTRUCTION = '拿起茶几上的刷子，把污渍擦干净，然后把刷子放回原处。'
TOOL = '/World/OpenEndWipe/Brush'
STAIN = '/World/OpenEndWipe/Stain'
SUPPORT = '/World/Assets/CoffeeTable/Mesh'


class WipeRegistration:
    asset = 'wipe_tool_hand_brush'
    selection_mode = 'contact'
    physics_hz = 60
    supported_physics_hz = (60, 240)
    initial_workspace = 'table_cloth'

    @staticmethod
    def validate_parameters(parameters):
        if parameters != {}:
            raise ValueError('Wiping uses scene binding; no task parameter overrides')

    @staticmethod
    def resolve_object(parameters, manifest):
        WipeRegistration.validate_parameters(parameters)
        group = manifest.get('wipe_table', {})
        if (group.get('tool_path') != TOOL or group.get('stain_path') != STAIN
                or group.get('support_path') != SUPPORT):
            raise ValueError('Missing exact brush/stain/support binding')
        rows = [o for o in manifest['objects'] if o['id'] == WipeRegistration.asset]
        if len(rows) != 1 or rows[0]['path'] != TOOL or rows[0].get('task_role') != 'wipe_tool':
            raise ValueError('Exactly one generated dynamic brush required')
        stains = [o for o in manifest['objects'] if o['id'] == 'table_stain_grid']
        if len(stains) != 1 or stains[0]['path'] != STAIN:
            raise ValueError('Exactly one generated stain grid required')
        layout = Path(group['layout_file'])
        if not layout.is_file() or hashlib.sha256(layout.read_bytes()).hexdigest() != group['layout_sha256']:
            raise ValueError('Stain layout source changed or missing')
        if group.get('randomization') is not None:
            from .random_stain import verify_generated
            verify_generated(group)
        return rows[0]
