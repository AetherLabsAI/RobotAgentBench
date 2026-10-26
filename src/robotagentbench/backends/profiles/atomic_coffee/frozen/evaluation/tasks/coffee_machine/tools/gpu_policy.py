"""Which GPU indices this flow may take on the current host - a scheduling policy, never a physics or scoring input.

User 2026-10-06 (202): no card is reserved any more (0, 1, 4, 5 included). Idle cards are taken preemptively, with one
courtesy rule: when MORE than two cards are idle, two of them are left idle for other people and only the rest are taken;
when two or fewer are idle, all of them may be taken. "Taking" a card only ever means launching on a card that is idle
(no compute process, no memory / load); a running job of anyone else is never stopped, evicted or shared.

ARENA_RESERVED_GPUS can still pin cards out explicitly (e.g. a host-specific rule); unset means no reserved card.
ARENA_IDLE_RESERVE overrides the number of idle cards left free (default 2).
"""
import os

DEFAULT_RESERVED = ()
VARIABLE = 'ARENA_RESERVED_GPUS'
IDLE_RESERVE_VARIABLE = 'ARENA_IDLE_RESERVE'
DEFAULT_IDLE_RESERVE = 2


def reserved():
    raw = os.environ.get(VARIABLE)
    if raw is None:
        return set(DEFAULT_RESERVED)
    return {int(value) for value in raw.replace(',', ' ').split()}


def idle_reserve():
    raw = os.environ.get(IDLE_RESERVE_VARIABLE)
    return DEFAULT_IDLE_RESERVE if raw in (None, '') else max(0, int(raw))


def takeable(idle):
    """Idle card indices this flow may take now: all of them when at most `idle_reserve()` are idle, otherwise all but
    that many (the lowest-numbered idle cards are the ones left free)."""
    idle = sorted(int(i) for i in idle if int(i) not in reserved())
    keep = idle_reserve()
    return idle if len(idle) <= keep else idle[keep:]


def check(*gpus):
    """Raise if any requested card is reserved on this host."""
    blocked = sorted(set(int(g) for g in gpus) & reserved())
    if blocked:
        raise ValueError(f'GPU {blocked} reserved on this host; set {VARIABLE} to change the policy')
    return tuple(int(g) for g in gpus)


def describe():
    return dict(reserved=sorted(reserved()), variable=VARIABLE, idle_reserve=idle_reserve(),
                rule='take idle cards; when more than idle_reserve cards are idle leave idle_reserve of them free (user 2026-10-06)',
                source='environment' if os.environ.get(VARIABLE) is not None else 'default (no reserved card, user 2026-10-06)')
