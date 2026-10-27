"""Keep public information across legacy code-tool observation filters.

The frozen broker still owns tool dispatch and execution. This outer boundary
only restores allowlisted context and map fields after its robot-only filter.
It neither renders cameras nor calls observe to supplement an action reply.
"""
from copy import deepcopy

from robotagentbench.interfaces.observation_extras import public_context, public_navigation


class ObservationHost:
    def __init__(self, host, *, context, navigation):
        self._host = host
        self._context = public_context(context)
        self._navigation = navigation

    def __getattr__(self, name):
        return getattr(self._host, name)

    def _observation(self, row):
        row = deepcopy(row)
        row['context'] = self._context
        if self._navigation is not None:
            row['navigation'] = public_navigation(self._navigation(deepcopy(row)), row['step'])
        return row

    def _reply(self, reply):
        reply = deepcopy(reply)
        if 'observation' in reply:
            before = (self.session.clock.current_time, self.session.control_index)
            reply['observation'] = self._observation(reply['observation'])
            if before != (self.session.clock.current_time, self.session.control_index):
                raise RuntimeError('Public information capture advanced the control clock')
        if reply.get('op') == 'ready':
            reply['contract']['observation_fields'] = list(reply['observation'])
        return reply

    def ready(self):
        return self._reply(self._host.ready())

    def observe(self):
        return self._reply(dict(observation=self._host.observe()))['observation']

    def exchange(self, request):
        return self._reply(self._host.exchange(request))

    def close(self):
        return self._host.close()
