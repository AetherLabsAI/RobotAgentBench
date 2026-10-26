"""Contact starts a persistent linear process; workspace remains independent.

Existing 20 Hz control observations are also the final-state recheck schedule.
No timers, solver rates, scene transforms or contact facts are fabricated.
"""
class ContactLifecycle:
    def __init__(self):self.started={}

    def activity(self,requests,ports):
        result={k:dict(v) for k,v in requests.items()}
        for key,value in result.items():
            if value.get('active') and value.get('last_touch_s') is not None:
                self.started.setdefault(key,value['last_touch_s'])
            if key in self.started and not ports[key].ended:
                value.update(active=True,locked=False)
                if 'last_touch_s' in value and value['last_touch_s'] is None:
                    value['last_touch_s']=self.started[key]
        return result
