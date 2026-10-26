"""Grasp hold latch (user 2026-10-06: "允许抓住后抖动").

Confirmation is unchanged: the public GraspDetector needs bilateral pad contact AND the target staying within 3 mm / 5 deg of the
hand for 0.1 s. Once a hand's grasp is confirmed it stays held while that hand's two pads keep touching the target with opposing
normals; relative jitter no longer breaks it. A loss of bilateral contact shorter than GAP_S is bridged, so a single-step pad
flicker does not drop the grasp. Pure Python, shared by the Isaac reader and the CPU rescoring tool.
"""
GAP_S = 0.1


class GraspLatch:
    def __init__(self, hands, physics_dt, gap_s=GAP_S):
        self.gap_steps = int(round(gap_s / physics_dt))
        self.held = {h: False for h in hands}
        self.gap = {h: 0 for h in hands}

    def update(self, confirmed, bilateral):
        """confirmed / bilateral: {hand: bool} of this physics step. Returns {hand: held}."""
        for h in self.held:
            if confirmed.get(h):
                self.held[h], self.gap[h] = True, 0
            elif self.held[h]:
                if bilateral.get(h):
                    self.gap[h] = 0
                else:
                    self.gap[h] += 1
                    if self.gap[h] > self.gap_steps:
                        self.held[h], self.gap[h] = False, 0
        return dict(self.held)
