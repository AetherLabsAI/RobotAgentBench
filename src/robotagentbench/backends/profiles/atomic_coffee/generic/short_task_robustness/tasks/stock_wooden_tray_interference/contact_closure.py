"""Per-digit stop-on-contact for the external actor, never robot scoring."""
import math

DIGITS=('index','middle','ring','pinky','thumb')


def digit_for_bone(name):
    if name.startswith('thumb.'):return 'thumb'
    for digit in DIGITS[:-1]:
        if name.startswith('f_'+digit+'.'):return digit
    return None


class ContactClosure:
    def __init__(self):
        self.caps={}
        self.last={d:0. for d in DIGITS}
        self.events=[]

    def command(self,desired,phase):
        if not math.isfinite(desired) or not 0<=desired<=1:raise ValueError('Invalid closure')
        self.last={d:(self.caps.get(d,1.)*desired if phase in ('release','retreat','done')
                      else min(desired,self.caps.get(d,1.))) for d in DIGITS}
        return dict(self.last)

    def observe(self,digits,phase,time_s):
        if any(d not in DIGITS for d in digits):raise ValueError('Unknown contact digit')
        if phase!='approach':return
        for d in digits:
            if d not in self.caps:
                self.caps[d]=self.last[d]
                self.events.append(dict(digit=d,closure=self.last[d],time_s=time_s,
                                        evidence='native_target_contact'))

    def ready(self,current_digits):
        # Opposed measured contacts, not palm-only touch or a close command.
        return 'thumb' in current_digits and any(d in current_digits for d in DIGITS[:-1])
