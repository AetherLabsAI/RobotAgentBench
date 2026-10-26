"""Reuse native object-transfer evidence and the unchanged OpenEnd bag logic."""
from openend.tasks.all_objects_integration.ports import ObjectPort


class UmbrellaPort(ObjectPort):
    asset = 'umbrella'
    key = 'umbrella_to_bag'
    mode = 'bag'
