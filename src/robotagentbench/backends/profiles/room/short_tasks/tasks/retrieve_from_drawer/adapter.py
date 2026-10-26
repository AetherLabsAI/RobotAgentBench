"""Closed drawer -> one wallet/glasses -> explicit bag/basket; one score.

Opening is an unscored prerequisite, using the existing slide's approved
10 cm open / 3 cm closed thresholds. Source clearance retains the approved
9-1/9-2 whole-collider-above-current-rim predicate, never a new AABB success.
"""
from dataclasses import replace
import json
import math
from short_tasks.tasks.pack_object.adapter import (
    request_parts, plan_base, unavailable, bind as bind_transfer,
    normalize, adapt_summary, ReleaseEvidencePort)
from evaluation.tasks.tidy_sofa.specs import DRAWER_OPEN_M, DRAWER_CLOSED_M

RULE_VERSION = "openend.short_task.retrieve_from_drawer.v1"


def resolve(request):
    options, _, _ = request_parts(request, "retrieve_from_drawer", RULE_VERSION,
        ("asset", "destination", "close_drawer"))
    asset = options.get("asset", "wallet")
    destination = options.get("destination", "fixed_cpu_bag")
    close = options.get("close_drawer", False)
    if not isinstance(asset, str) or not isinstance(destination, str) or type(close) is not bool:
        raise ValueError("Asset/destination strings and explicit boolean close_drawer required")
    component = "glasses_transfer" if asset == "glasses_case" else "wallet_transfer"
    label = "sunglasses case" if asset == "glasses_case" else "wallet"
    dest_label = "item-storage basket beside the trash bins" if destination == "storage_basket" else "fixed open tote bag"
    instruction = "Open the coffee table drawer, take out the %s, place it inside the %s, and release it. Closing the drawer afterward is not required." % (label, dest_label)
    if close:
        instruction = "Open the coffee table drawer, take out the %s, place it inside the %s, release it, and finally close the drawer." % (label, dest_label)
    plan = plan_base(request, "retrieve_from_drawer", RULE_VERSION, component, asset,
        "coffee_table.drawer", destination, 7, instruction,
        ["One scene-owned dynamic slide drawer, initially closed: actual calibrated travel <= 0.03 m.",
         "Exactly one target actually in and supported by this drawer, initially unheld.",
         "Existing handle, slide, body frame calibration and selected destination must bind uniquely.",
         "Opening travel >= 0.10 m must occur after initial admission before pickup progresses.",
         "Source separation: lowest target collider above highest CURRENT drawer wall collider.",
         "Open drawer is an unscored prerequisite; approved pickup 3 + placement process 2 + current final 2.",
         "close_drawer=false is explicit: no hidden final drawer-close prerequisite."])
    plan["close_drawer"] = close
    plan["evidence_policy"].update(drawer_open_m=DRAWER_OPEN_M,
        drawer_closed_m=DRAWER_CLOSED_M,
        drawer_open_points=0, source_clearance="whole_target_collider_above_current_drawer_rim",
        drawer_initial_pose_writes=0)
    if asset not in ("wallet", "glasses_case"):
        return unavailable(plan, "unsupported_drawer_target", "首版抽屉来源只支持钱包和墨镜盒。")
    if destination not in ("fixed_cpu_bag", "storage_basket"):
        return unavailable(plan, "unsupported_drawer_destination", "抽屉任务只接入固定开口包或物品收纳篮。", "load")
    if close:
        return unavailable(plan, "close_drawer_combination_not_frozen", "首版冻结为close_drawer=false；取物后关抽屉的新组合尚未批准。")
    return plan


def bind(plan, manifest):
    rows = bind_transfer(plan, manifest)
    if not manifest.get("openend_drawer"):
        raise ValueError("Unified scene must provide native drawer-frame calibration")
    return rows


class DrawerPrerequisite:
    """Only unscored causal opening; no synthetic pose/score/stage resets."""
    def __init__(self):
        self.accepted = False
        self.opened = False

    def accept(self, initial_travels, inside, supported):
        if self.accepted:
            raise ValueError("Initial drawer admission occurs exactly once")
        if not initial_travels or any(type(v) not in (int, float) or not math.isfinite(v) for v in initial_travels):
            raise ValueError("Measured drawer history required")
        if any(v > DRAWER_CLOSED_M for v in initial_travels):
            raise ValueError("Drawer is not initially closed")
        if type(inside) is not bool or type(supported) is not bool:
            raise ValueError("Measured drawer containment and support booleans required")
        if not inside:
            raise ValueError("Target collider extends outside selected drawer; see drawer_initial_diagnostic.json")
        if not supported:
            raise ValueError("Target lacks continuous exclusive drawer support; see drawer_initial_diagnostic.json")
        self.accepted = True

    def observe(self, travel):
        if type(travel) not in (int, float) or not math.isfinite(travel):
            raise ValueError("Measured drawer travel required")
        if self.accepted and travel >= DRAWER_OPEN_M:
            self.opened = True
        return self.opened


class DrawerEvidencePort(ReleaseEvidencePort):
    def __init__(self, stage, manifest, *args, **kwargs):
        super().__init__(stage, manifest, *args, **kwargs)
        if not self.drawer:
            raise ValueError("Closed-drawer Short Task requires actual drawer source")
        import numpy as np
        calibration = stage.GetPrimAtPath(self.drawer).GetCustomDataByKey("openend:drawerCalibration")
        if not calibration:
            raise ValueError("Scene-owned drawer calibration required")
        self.short_calibration = json.loads(calibration)
        self.short_rotation = np.asarray(self.short_calibration["source_to_layout_matrix"], dtype=float)[:3, :3]
        if not np.allclose(self.short_rotation @ self.short_rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("Rigid drawer frame calibration required")
        self.short_drawer = DrawerPrerequisite()

    def after_physics(self, helper_report):
        super().after_physics(helper_report)
        import numpy as np
        t = np.asarray(self.drawer_view.get_transforms()).copy()[0]
        travel = float(np.dot(t[:3]-self.short_calibration["initial_origin_m"], self.short_rotation[0])) + self.short_calibration["initial_lip_m"]
        self.latest["_short_drawer_travel_m"] = travel
        self.latest["_short_drawer_opened"] = self.short_drawer.observe(travel)

    def accept_initial(self):
        from openend.tasks.meal_tray.nonmotion_closeout_v1.initial_windows import initial_window
        from evaluation.tasks.find_object.isaac_evidence import body_colliders
        from evaluation.tasks.shelf_pick.geometry import rotation
        import numpy as np
        rows, complete = initial_window(self.history, self.config.initial_history_s, self.clock.get_physics_dt())
        if not complete:
            raise ValueError("Full native initial drawer history required")
        drawer_pose = np.asarray(self.drawer_view.get_transforms()).copy()[0]
        drawer_q = tuple(drawer_pose[[6, 3, 4, 5]])
        pose = self.latest["pose"]
        world = rotation(pose.quaternion_wxyz).apply(self.target_colliders) + pose.position_m
        local = rotation(drawer_q).inv().apply(world-drawer_pose[:3])
        container = np.vstack(list(body_colliders(self.stage, self.drawer).values()))
        # Same initial-fit policy as the upstream drawer source sampler: XY fit,
        # no floor underrun, original source rim protrusion is retained.
        inside = bool(np.all(local.min(0) >= container.min(0)) and
            np.all(local.max(0)[:2] <= container.max(0)[:2]))
        supported = all(bool(r["supports"]) and all(p == self.drawer or p.startswith(self.drawer+"/")
            for p in r["supports"]) for r in rows)
        diagnostic = dict(inside=inside, supported=supported, drawer=self.drawer,
            target_body=self.geometry.body, complete_history=complete,
            target_local_bounds=[local.min(0).tolist(), local.max(0).tolist()],
            drawer_local_bounds=[container.min(0).tolist(), container.max(0).tolist()],
            lower_clearance_mm=((local.min(0)-container.min(0))*1000).tolist(),
            upper_xy_clearance_mm=((container.max(0)[:2]-local.max(0)[:2])*1000).tolist(),
            position_span_mm=(np.ptp(np.array([r["pose"].position_m for r in rows]),axis=0)*1000).tolist(),
            history=[dict(time_s=r["time_s"], supports=sorted(r["supports"]),
                travel_m=r["_short_drawer_travel_m"], robot_contact=bool(r["robot_contact"]),
                on_floor=bool(r["on_floor"])) for r in rows], thresholds_changed=False)
        (self.output/"drawer_initial_diagnostic.json").write_text(json.dumps(diagnostic, indent=2)+"\n")
        self.short_drawer.accept([r["_short_drawer_travel_m"] for r in rows], inside, supported)
        receipt = super().accept_initial()
        receipt = dict(receipt, drawer_closed=True, target_in_drawer=True,
            drawer_open_points=0, drawer_close_required=False,
            clearance_predicate="whole_target_collider_above_current_drawer_rim")
        (self.output/"short_initial_acceptance.json").write_text(json.dumps(receipt, indent=2)+"\n")
        return receipt

    def facts(self, row, active, index):
        plain = {k: v for k, v in row.items() if not k.startswith("_short_")}
        frame = super().facts(plain, active, index)
        # No early grasp/lift rewards can precede the required actual opening.
        return replace(frame, active=bool(active and row.get("_short_drawer_opened", False)))

    def _short_summary(self):
        self.score = adapt_summary(self.composite.summary(), self.composite.last,
            require_drawer=True, drawer_opened=self.short_drawer.opened)
        self.score["evidence_status"] = "native_physx"
        return self.score


def factory_override(plan, declaration):
    if declaration not in plan["runtime_components"] or not plan["available"]:
        raise ValueError("Available resolved drawer declaration required")
    from openend.tasks.scene_requirements.runtime_ports import WalletPort, GlassesPort
    cls = WalletPort if declaration["type"] == "wallet_transfer" else GlassesPort
    return type("ShortDrawer"+cls.__name__, (DrawerEvidencePort, cls), {})
