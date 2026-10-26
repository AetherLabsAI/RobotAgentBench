"""Approved sofa recovery destination: the actual whole coffee-table top.

This task-local evidence adapter changes only surface containment. It reuses
the original native object/contact reader, pickup/destination scorers, support
identity, release checks, timing, and all physical parameters.
"""
from dataclasses import replace

RULE_VERSION = 'short_task_robustness.grasp_failure_recovery.sofa_whole_table.v1'
INSTRUCTION = ('Pick up the TV remote control from the sofa, place it on the '
               'coffee table, and release it.')


def within_table(footprint, faces):
    """Original measured face-union predicate, without the reserved subregion."""
    from openend.tasks.remote_transfer.geometry import inside, polygon
    target = polygon(footprint)
    surfaces = tuple(polygon(face) for face in faces)
    if not surfaces:
        raise ValueError('Measured coffee-table top faces required')
    return all(any(inside(point, face) for face in surfaces) for point in target)


def plan_metadata(plan):
    if plan.get('asset') != 'remote_tv' or plan.get('task_name') not in (
            'sofa_to_coffee_table', 'failure_recovery_sofa'):
        raise ValueError('Only sofa TV-remote failure recovery')
    plan.update(instruction=INSTRUCTION, rule_version=RULE_VERSION,
                destination='coffee_table.whole_surface',
                placement_policy_version=RULE_VERSION,
                initial_requirements=['unique_scene_owned_sofa_target', 'actual_coffee_table_top'])
    return plan


def port_class():
    # Lazy heavy dependency: CPU metadata/plan queries never import Isaac.
    from short_tasks.tasks.sofa_to_drawer.native_port import CombinationPort

    class WholeTableRemotePort(CombinationPort):
        asset = 'remote_tv'
        mode = 'surface'
        key = 'remote_tv_to_coffee_table'

        def facts(self, row, active, index):
            original = super().facts(row, active, index)
            geometry = self.geometry
            measured_inside = within_table(geometry.footprint(row['pose']), geometry.top['faces'])
            return replace(original, in_zone=measured_inside)

    return WholeTableRemotePort
