"""Keep all legacy validation except grasp-implies-bilateral-pad-contact.

Only Openend's process subclasses use this mixin. Standalone source is unedited.
Real bilateral contacts must STILL imply real robot contact.
"""
from evaluation.tasks.air_fryer import process_rules as old


def validate_cake(name, obj):
    if not isinstance(obj, old.CakeFacts):
        raise ValueError('Cake facts required: ' + name)
    old._pose(name, obj)
    for key in ('pads_bilateral', 'grasped', 'robot_contact', 'ground_contact',
                'basket_contact', 'inside_basket', 'outside_basket'):
        old._flag(name + '.' + key, getattr(obj, key))
    if not isinstance(obj.supports, (frozenset, set)) or not obj.supports <= old.SURFACES:
        raise ValueError('Unknown support surface: ' + name)
    if obj.pads_bilateral and not obj.robot_contact:
        raise ValueError('Finger pads touch but the robot does not: ' + name)
    if old.GROUND in obj.supports and not obj.ground_contact:
        raise ValueError('Ground supports without ground contact: ' + name)
    if (old.BASKET_FLOOR in obj.supports or old.BASKET_WALL in obj.supports) and not obj.basket_contact:
        raise ValueError('Basket supports without basket contact: ' + name)
    if obj.inside_basket and obj.outside_basket:
        raise ValueError('A cake cannot be inside and outside the basket at once: ' + name)


class FollowValidation:
    def _validate(self, f):
        if not isinstance(f, old.Facts):
            raise ValueError('Facts of one control frame required')
        expected = 0 if self._last_step is None else self._last_step + 1
        if type(f.step) is not int or f.step != expected:
            raise ValueError('Skipped, repeated or out-of-order control frame')
        old._number('time_s', f.time_s)
        if abs(f.time_s - f.step * old.CONTROL_DT_S) > old.CLOCK_TOLERANCE_S:
            raise ValueError('time_s is not on the 20 Hz control clock')
        validate_cake('target', f.target)
        validate_cake('other', f.other)
        old._tray('tray', f.tray)
