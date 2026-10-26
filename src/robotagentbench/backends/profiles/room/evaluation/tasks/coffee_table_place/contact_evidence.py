"""Reuse dining_place's contact reduction, rebinding identities only.

Paths below are private measurement roles, never USD edits or collision filters.
An unselected physical plate/payload MUST NOT accidentally match a logical role.
This module produces contact facts, not region membership, scores or success.
"""
from .bindings import PlacementBinding


class PlacementContacts:
    def __init__(self, binding, resolve, sleeping, event_types, surface_top_m, floor_paths=()):
        from evaluation.tasks.dining_place.isaac_detection import (
            ContactStream, PLATE, PAYLOAD, DESTINATION, ROBOT, under)
        if not isinstance(binding, PlacementBinding):
            raise TypeError('Explicit task-owned placement binding required')
        self.binding = binding
        self._held_role, self._payload_role = PLATE, PAYLOAD
        self._robot_role = ROBOT
        self._under = under
        # 落地事实（用户 2026-09-23，沿用 4-x 做法）：目标与实际地面碰撞体发生接触。
        # 地面保持独立角色，不并入目的地，以免把“掉到地上”读成“已放入”。
        self._floor_role = '/Floor'
        self.floor_paths = tuple(floor_paths)
        mapping = [(binding.held_body, PLATE), (binding.destination, DESTINATION),
                   (binding.robot, ROBOT)] + [(p, self._floor_role + f'/{i}')
                                              for i, p in enumerate(self.floor_paths)]
        if binding.payload is not None:
            mapping.append((binding.payload, PAYLOAD))

        def role_path(token):
            path = str(resolve(token))
            for actual, role in mapping:
                if under(path, actual):
                    return role + path[len(actual):]
            # Prefix prevents unrelated scene instances named MicrowaveTakePlate,
            # MicrowaveBeefbox or CounterTop_17 from being mistaken for the target.
            return '/Unbound' + path

        self.stream = ContactStream(role_path, sleeping, event_types, surface_top_m)

    def read(self, report, held_up):
        reduced = self.stream.read(report, held_up)
        external_support = False
        on_floor = False
        for key,item in self.stream.active.items():
            if key in self.stream.unknown_pairs:continue
            paths, _, _, _, _, up0, up1, _, _ = item
            for i in (0, 1):
                if not self._under(paths[i], self._held_role):
                    continue
                other = paths[1-i]
                if self._under(other, self._floor_role):
                    on_floor = True
                if not any(self._under(other, root) for root in
                           (self._robot_role, self._held_role, self._payload_role)):
                    external_support |= up0 if i == 0 else up1
        result = dict(
            held_grasped=reduced['plate_grasped'],
            held_supported_by_destination=reduced['plate_on_table'],
            held_on_external_support=bool(external_support),
            held_on_floor=bool(on_floor),
            reported_point_reads=reduced['reported_point_reads'],
            active_pair_count=reduced['active_pair_count'],
            sleep_cached_pairs=reduced['sleep_cached_pairs'],
            contact_evidence=reduced['contact_evidence'],
        )
        if self.binding.payload is not None:
            result.update(payload_grasped=reduced['payload_grasped'],
                payload_supported_by_held_body=reduced['payload_supported_by_plate'],
                payload_on_other_support=reduced['payload_on_other_support'])
        return result
