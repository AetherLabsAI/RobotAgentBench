# -*- coding: utf-8 -*-
"""Legacy recording/runtime adapter catalog used by teleop reproduction.

This is not a task registry.  Benchmark definitions and task identity live in
``real2sim.production.benchmark_eval_registry``.  Numeric keys here are old
recording labels accepted by legacy initializer CLIs; benchmark-keyed entries
are implementation adapters.  Either kind may supply source data or geometry,
but neither can redefine a benchmark task.  Each entry:

    family   : pick | place | throw | snack | tier2 | cloth   (which tuned init handles placement)
    key      : the argument that family's init wants (task_id for most; a NAME for snack)
    root     : USD parent path of the target prim (for track_prim = root/<target>)
    distract : distractor prim NAMES under root that stay as immovable scenery (measure gate)
    new      : (compound_task_N, subtask_idx) link into the 10-task output taxonomy
    zh       : the object, for humans
    required_count / probe_scope / preferred_probes : optional full-task
             cardinality and explicitly partial single-object probe metadata
    required_evidence_contract : optional publication contract; runners must
             not claim it unless the produced sidecar validates against it
    contact_context : required non-target prims in direct support, resting on,
             clamping, occluding, or containment contact with the primary
             object.  These are task scene content, not optional visual
             distractors.
    contact_context_scope : must be ``per_episode`` when contact_context is
             non-empty.  Each episode needs its own reviewed pose, geometry,
             collision, and dynamics evidence; one episode's placement must
             never be reused as a task-family default.
    episode_uids : required on every required per-episode contact item.  It
             binds that exact secondary-object identity to the reviewed raw
             episode(s); entries without a matching UUID fail closed.

track_prim and distract_prims are DERIVED here so downstream code needs zero family knowledge.

Retail availability is deliberately separate from task identity.  Historical
entries remain registered, while shelf_asset_policy.py marks unsafe/stale
models fail-closed at every execution entrypoint.
"""

from shelf_asset_policy import task_asset_decision, usable_prims
from bag_place_contract import BAG_PLACE_TARGETS

SHELF = "/World/Assets/ShelfItems"
TIER2 = "/World/Assets/Tier2Items"
ASSETS = "/World/Assets"

# Only pinned refined items may remain as shelf scenery/distractors.  The
# policy-generated scene filter deactivates every other direct child.
_SNACK_DIST = list(usable_prims(SHELF))
_TIER2_GRID = list(usable_prims(TIER2))

REGISTRY = {
    # --- snack drinks: pick from shelf (init_snack, key = name) ---
    "6":  dict(family="snack", key="orange g", root=SHELF, zh="橙色运动饮料", new=(3, 3),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "8":  dict(family="snack", key="coke",     root=SHELF, zh="可乐",         new=(3, 13),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "9":  dict(family="snack", key="blue g",   root=SHELF, zh="蓝色运动饮料", new=(3, 5),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "10": dict(family="snack", key="sprite",   root=SHELF, zh="雪碧",         new=(3, 15),
               required_evidence_contract="real2sim.pick_contact.v1"),
    # Newer Sprite recording uses the same document-approved 355 ml model.
    "21": dict(family="snack", key="sprite",   root=SHELF, zh="雪碧（新录制）", new=(3, 15),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "22": dict(family="snack", key="blue g",   root=SHELF, zh="蓝色运动饮料（新录制）", new=(3, 5),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "48": dict(family="snack", key="starbucks", root=SHELF, prim="starbucks",
               zh="含咖啡因饮料（Starbucks）", new=(3, 7),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "50": dict(family="snack", key="sparkling", root=SHELF,
               prim="grapefruit_can", zh="气泡水", new=(3, 11),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "51": dict(family="snack", key="red bull", root=SHELF,
               prim="red_bull_pink", zh="含咖啡因饮料（Red Bull）",
               # Owner decision 2026-08-07: benchmark 3.7 is the Starbucks
               # variant (small Task 48).  Keep this robot template runnable
               # without claiming it is the benchmark's canonical product.
               new=None,
               required_evidence_contract="real2sim.pick_contact.v1"),
    "85": dict(family="snack", key="beer", root=SHELF,
               prim="cayman_jacked", zh="酒精饮料", new=(3, 1),
               required_evidence_contract="real2sim.pick_contact.v1"),
    # --- snack drinks: place on table (init_place) ---
    "14": dict(family="place", key="14", root=SHELF, prim="gatorade_m", zh="橙饮→茶几", new=(3, 4)),
    "15": dict(family="place", key="15", root=SHELF, prim="gatorade_t", zh="蓝饮→茶几", new=(3, 6)),
    "16": dict(family="place", key="16", root=SHELF, prim="gatorade_m", zh="橙饮→茶几", new=(3, 4)),
    "17": dict(family="place", key="17", root=SHELF, prim="gatorade_t", zh="蓝饮→茶几", new=(3, 6)),
    "18": dict(family="place", key="18", root=SHELF, prim="sprite_c1",  zh="雪碧→茶几", new=(3, 16)),
    "19": dict(family="place", key="19", root=SHELF, prim="diet_coke",  zh="健怡可乐→茶几", new=(3, 14)),
    "20": dict(family="place", key="20", root=ASSETS, prim="Book",      zh="书→茶几",   new=(6, 2)),
    "27": dict(family="place", key="27", root=ASSETS, prim="Book",      zh="粉书→茶几", new=(6, 2)),
    # Benchmark-only code-control identities.  These are deliberately not
    # robot-template small-task IDs: each action is the physical time reverse
    # of the named real pick from the exact requested destination.
    "benchmark_tidy_sofa_6_3": dict(
        family="place", key="benchmark_tidy_sofa_6_3", root=ASSETS,
        prim="Book", zh="benchmark 6.3 书→小茶几",
        benchmark_subtask="6.3", real_data_reference_id="60", new=None,
    ),
    "benchmark_tidy_sofa_6_5": dict(
        family="place", key="benchmark_tidy_sofa_6_5", root=ASSETS,
        prim="Note", zh="benchmark 6.5 绿色账单→茶几抽屉",
        benchmark_subtask="6.5", real_data_reference_id="28", new=None,
        action_relation=(
            "small_Task_28_identity_with_reviewed_two_hand_drawer_"
            "choreography_reference"),
    ),
    "benchmark_tidy_sofa_6_7": dict(
        family="place", key="benchmark_tidy_sofa_6_7", root=ASSETS,
        prim="BaseballCap", zh="benchmark 6.7 帽子→衣架",
        benchmark_subtask="6.7", real_data_reference_id="70", new=None,
        action_relation="direct_real_place_to_same_destination",
    ),
    "benchmark_tidy_sofa_6_9": dict(
        family="place", key="benchmark_tidy_sofa_6_9", root=ASSETS,
        prim="Remote", zh="benchmark 6.9 遥控器→茶几",
        benchmark_subtask="6.9", real_data_reference_id="61", new=None,
    ),
    "benchmark_tidy_sofa_6_10": dict(
        family="place", key="benchmark_tidy_sofa_6_10", root=ASSETS,
        prim="Remote", zh="benchmark 6.10 遥控器→电视柜",
        benchmark_subtask="6.10", real_data_reference_id="58", new=None,
    ),
    "benchmark_tidy_sofa_6_15": dict(
        family="place", key="benchmark_tidy_sofa_6_15", root=ASSETS,
        prim="Wallet", zh="benchmark 6.15 钱包→茶几抽屉",
        benchmark_subtask="6.15", real_data_reference_id="92", new=None,
    ),
    "86": dict(family="place", key="86", root=SHELF, prim="good2grow",
               zh="儿童果汁→茶几", new=(3, 10)),
    "87": dict(family="place", key="87", root=SHELF, prim="grapefruit_can",
               zh="西柚气泡水→茶几", new=(3, 12)),
    "88": dict(family="place", key="88", root=SHELF, prim="cayman_jacked",
               zh="Cayman Jacked→茶几", new=(3, 2)),
    "89": dict(family="place", key="89", root=SHELF, prim="starbucks",
               zh="Starbucks Espresso→茶几", new=(3, 8)),
    # --- tier2 shelf snacks: pick (init_tier2, key = id) ---
    "35": dict(family="tier2", key="35", root=TIER2, prim="chomps_r",     zh="牛肉棒",   new=(3, 19),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "36": dict(family="tier2", key="36", root=TIER2, prim="pring_bbq",    zh="薯片",     new=(3, 21),
               required_evidence_contract="real2sim.pick_contact.v1"),
    # The manually maintained Sheet reports a completed benchmark 3.25 sim
    # rollout.  Per project-owner direction on 2026-08-07, that accepted
    # rollout is the semantic authority for the historical "pick packs"
    # identity; keep the concrete Honey Maid asset name visible instead of
    # inventing a different hidden product.
    "37": dict(family="tier2", key="37", root=TIER2, prim="honeymaid",
               zh="Honey Maid packs（Sheet 3.25 人工确认）", new=None,
               required_evidence_contract="real2sim.pick_contact.v1"),
    "38": dict(family="tier2", key="38", root=TIER2, prim="skit_littles", zh="糖果",     new=(3, 29),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "39": dict(family="tier2", key="39", root=TIER2, prim="zbar_t",       zh="蛋白棒",   new=(3, 27),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "40": dict(family="tier2", key="40", root=TIER2, prim="oreo_gf",      zh="奥利奥",   new=(3, 23),
               required_evidence_contract="real2sim.pick_contact.v1"),
    "47": dict(family="snack", key="kid juice", root=SHELF, prim="good2grow", zh="儿童果汁", new=(3, 9),
               required_evidence_contract="real2sim.pick_contact.v1"),
    # --- tier2 snacks: place on tray/table (init_place) ---
    "41": dict(family="place", key="41", root=TIER2, prim="honeymaid", zh="Honey Maid→茶几", new=None),
    "42": dict(family="place", key="42", root=TIER2, prim="pring_bbq",    zh="薯片→茶几", new=(3, 22)),
    "43": dict(family="place", key="43", root=TIER2, prim="chomps_r",     zh="牛肉棒→茶几", new=(3, 20)),
    "44": dict(family="place", key="44", root=TIER2, prim="skit_littles", zh="糖果→茶几", new=(3, 30)),
    "45": dict(
        family="place", key="45", root=TIER2, prim="oreo_gf",
        zh="奥利奥→茶几（统一 shelf-to-CoffeeTable 机位）", new=(3, 24),
    ),
    "46": dict(family="place", key="46", root=TIER2, prim="zbar_t", zh="Z-Bar→托盘", new=(3, 28)),
    # small Tasks 71-74 share the task-local soft-tote scene imported from l2.
    # Each is still an ordinary physical place contract: one initially held
    # target, a collision-backed clamp, recorded release, and deformable-tote
    # settling.  Target identity remains task-specific.
    **{
        task_id: dict(
            family="place",
            key=task_id,
            root=target["root"],
            prim=target["prim"],
            zh=target["label"],
            new=None,
            **(
                {
                    "benchmark_segment_contracts": {
                        "5.11": "sofa_pick_segment_of_same_long_real_episode",
                        "5.12": "bag_place_segment_of_same_long_real_episode",
                    }
                }
                if task_id == "73" else {}
            ),
        )
        for task_id, target in BAG_PLACE_TARGETS.items()
    },
    # Historical L2 articulated air-fryer drawer-removal identities. The
    # obsolete DiningTableAirFryer v19 scene fixture was removed on 2026-08-08;
    # these records remain only to preserve task identity and now fail closed
    # until a replacement drawer asset and trajectory calibration are accepted.
    "66": dict(
        family="drawer",
        key="66",
        root=(
            "/World/Background/Shell/DiningTableAirFryer/Asset/Centered"
        ),
        prim="Tray",
        zh="small Task 66 (sugar cake out)",
        new=None,
        required_evidence_contract="real2sim.drawer_remove_trace.v1",
    ),
    "67": dict(
        family="drawer",
        key="67",
        root=(
            "/World/Background/Shell/DiningTableAirFryer/Asset/Centered"
        ),
        prim="Tray",
        zh="small Task 67 (no sugar cake out)",
        new=None,
        required_evidence_contract="real2sim.drawer_remove_trace.v1",
    ),
    # --- sofa tidy: pick note / book off the sofa seat (init_pick) ---
    # 2026-08-03: these three carried a station in task_objects.json, and task
    # 12 even carried contact physics in TASK_SIM_ENV_DEFAULTS, but none was
    # registered here. init_task looks the family up in this table, so the
    # uniform entry point could not dispatch them at all -- and the failure
    # surfaced as an unrelated USD warning, because init_task reports the
    # sub-initializer's stderr when it gets no JSON back. Half-registered is
    # worse than absent: they looked supported right up to the render.
    # Registering them does NOT claim they grasp; 12/13 are on the known
    # contact-physics list. It only makes them dispatchable and honest.
    "11": dict(
        family="pick", key="11", root=ASSETS, prim="Note",
        zh="便签@沙发", new=None,
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "12": dict(
        family="pick", key="12", root=ASSETS, prim="Book",
        zh="书@沙发", new=None,
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "13": dict(
        family="pick", key="13", root=ASSETS, prim="Book",
        zh="书@沙发(仅书)", new=None,
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    # --- pick / find (init_pick, key = id). OBJECT IDENTITY from robot dlb.db (2026-07-22):
    #     glasses<->wallet were swapped in earlier guesses.  Wallet now has a
    #     dedicated asset, but every pre-correction GlassesCase result remains invalid. ---
    "31": dict(
        family="pick", key="31", root=ASSETS, prim="Remote",
        zh="遥控器@沙发(controller)", new=(4, 13),
        # Freeze the first three manifest episodes already exercised by the
        # legacy runner.  The first is its sole geometric PASS; the next two
        # are deterministic failures.  None has target-only pad-force
        # evidence, so all three must be revalidated under contact-v1.
        preferred_probes=(
            "0855fa5b-d317-4374-ba83-f312379ac5aa",
            "0891eba5-9d51-42fe-bb34-5f0c9645d15e",
            "12464bec-7db6-4621-b397-2c0830a0efef",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31 (see
        # episode_support_context_audit_v1.json): these reviewed episodes
        # start with the target resting ON the listed asset.  Declared
        # non-required so the currently-accepted flat pool keeps rendering;
        # flip to required=True per item once its per-episode support
        # placement is calibrated.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="Book", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "12464bec-7db6-4621-b397-2c0830a0efef",
                    "3db4ff3b-4fd9-4dac-b350-3948c57b3a97",
                    "5f8e95f4-71e0-4a02-b812-6bc74cf8ddc2",
                )),
            dict(
                prim="Note", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "6c9c4437-6162-4199-a2b4-0c826510fc97",
                    "8a64115b-1c36-4d89-a84a-db5f2c6f50ca",
                    "bcd0e193-65f0-450a-9c5a-84169d3d0654",
                    "dbdd0d42-df47-40a8-ae1c-ad3a340cc336",
                )),
            dict(
                prim="CushionOnBook", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "32c3eb65-3a8c-4720-9188-b12026df824a",
                    "48931940-8e10-4247-994a-22ec9e0098e4",
                    "56e773e7-d1bc-4318-82d5-cc40d934792e",
                    "5be54fe5-8889-4c01-b4ad-d290bcf300fe",
                    "91da508c-3367-4c86-96a5-855a32854f39",
                    "b899db16-4c43-41e6-94c6-bd2f12e1c63b",
                    "c68d8036-3909-4358-8900-609a6d45423d",
                    "d2854fd1-ccba-4d92-a204-c619938fd0e1",
                    "e43b8601-d1e3-4789-b893-46c745117b55",
                    "ecb3ada5-fd90-467e-b6a5-620e05e3854d",
                    "ee8d9172-34bd-4d65-95c1-567756b6cf79",
                )),
        ),
    ),
    "32": dict(
        family="pick", key="32", root=ASSETS, prim="BaseballCap",
        zh="帽子@沙发(hat)", new=(6, 6),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31 (see
        # episode_support_context_audit_v1.json): these reviewed episodes
        # start with the target resting ON the listed asset.  Declared
        # non-required so the currently-accepted flat pool keeps rendering;
        # flip to required=True per item once its per-episode support
        # placement is calibrated.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="CushionOnBook", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "0a16d43f-8385-4430-93df-4a3f4500416a",
                    "157318fd-fd06-4966-9e6d-6d79b4eda7f9",
                    "25190243-0a32-444f-8bdb-6a759b2d4be8",
                    "2a3e4d23-9904-4a62-bf08-a20e8a3e02f8",
                    "2b94e73f-bc7a-4f73-a8cb-c9db7aa81fce",
                    "306f1e07-8105-436f-a620-18ca4d67559e",
                    "3a5867ef-4ef1-44b5-b3b1-4be68cc1e7d6",
                    "44598b6a-d4e0-4b33-8869-485a74565159",
                    "4bf1a15f-f93e-4030-ab3d-46945b0ff82e",
                    "538df803-9144-465a-83ec-cf3381f07517",
                    "5527d28c-169d-402f-8c64-49948b240b2f",
                    "700ddbfe-3c08-40b7-b573-ed2ca2438ca4",
                    "80bb4c5b-89ff-43a3-a95a-0bbd2d5cc228",
                    "88115408-8a23-4b08-940a-2796b6ffa6af",
                    "8e2c3997-2131-491f-9ad6-264211ef7cfb",
                    "aef098e3-f354-47e5-8af0-c21d8f45227b",
                    "b73264b2-b41d-4505-9284-f694186570bd",
                    "bcce4f87-05fd-4144-b95e-1c15f25991e3",
                    "c0dd154c-cdd1-436b-b4bb-9f54849d4daa",
                    "c9b37e3e-5020-4417-aabb-b978288debfe",
                    "d602fe4b-b452-42aa-9d24-8e8c1198259e",
                    "d83e1626-88ba-41d6-a97e-741cd00bb30e",
                    "f0f746ae-172b-44c9-a8e0-2cf7d5d3a60d",
                    "fb413e7e-6d07-4bf0-a7ba-56435ef4fd0b",
                )),
            dict(
                prim="PaperCup", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "f521a5e3-a0b5-4fc9-94fa-be38abf60792",
                )),
        ),
    ),
    "33": dict(
        family="pick", key="33", root=ASSETS, prim="PaperCup",
        zh="纸杯@沙发", new=(6, 11),
        # The legacy geometric runner reported clean sustained carries for
        # these three right-hand episodes, but those files contain no
        # target-only pad contact evidence.  Pin the lowest-pregrasp-slide
        # successes for a bounded contact-v1 revalidation instead of treating
        # the historical PASS strings as publishable evidence.
        preferred_probes=(
            "3413671a-a037-4c6f-8e8a-c1c23ce7f838",
            "377ca3df-616f-4625-80c7-9729b163df07",
            "1c56aac2-877d-4e62-8cf4-4abf06074db7",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31 (see
        # episode_support_context_audit_v1.json): these reviewed episodes
        # start with the target resting ON the listed asset.  Declared
        # non-required so the currently-accepted flat pool keeps rendering;
        # flip to required=True per item once its per-episode support
        # placement is calibrated.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="Book", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "000cb4ae-7f49-4177-80f0-8a37aa9ff79c",
                    "1082532a-3322-4132-aa1a-31a9659da749",
                    "1119e4f6-c667-459c-b5ea-b51e3daa8527",
                    "1c56aac2-877d-4e62-8cf4-4abf06074db7",
                    "38a098d7-d17e-4573-928a-a0c7fe667d98",
                    "51a75939-e068-4e37-a594-074858980059",
                    "81832b70-5c33-4ec9-9c82-f9f46d9060c9",
                    "8642473e-9e79-4909-b72e-56f270fb2628",
                    "af403c85-fe5c-4d4e-8e8e-297f362a3fea",
                    "c088b523-7d57-4d59-a21d-a7fd6964f9b5",
                )),
            dict(
                prim="Note", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "4e18d157-c694-4204-a914-690f353efc5a",
                    "a22cd642-b9f2-45b0-92f3-aa8326baa9f3",
                )),
            dict(
                prim="CushionOnBook", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "3413671a-a037-4c6f-8e8a-c1c23ce7f838",
                    "4b8cde65-0e24-4c03-999b-60e44f7b903c",
                    "b184b824-264d-46e1-98a8-b185f5b77ac5",
                    "714948ff-57bc-4ba0-b821-25a1f2d714e1",
                )),
        ),
    ),
    "53": dict(
        family="pick", key="53", root=ASSETS, prim="Wallet",
        zh="钱包@沙发", new=(4, 7),
        preferred_probes=(
            "7217cfe2-16a4-4b14-b82f-f09bd65cb9dc",
            "f913c9e0-5e5c-4f64-98d4-7b046de95453",
            "1b241e0d-33d4-4d46-9eb4-09a788370c02",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "54": dict(
        family="pick", key="54", root=ASSETS, prim="Umbrella",
        zh="伞@沙发", new=(5, 1),
        # Freeze the first three manifest episodes already exercised by the
        # legacy runner.  Its first row is a geometric PASS and the next two
        # are deterministic failures, but none contains target-only pad
        # forces; all remain diagnostic until contact-v1 revalidation.
        preferred_probes=(
            "016df0b3-8fce-4c2f-bf7f-68347f74bddd",
            "0ef2aa03-0ddf-4fbb-af4f-5075cc301e63",
            "156b77ad-32b6-4e5c-8c9c-373c1184b998",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31 (see
        # episode_support_context_audit_v1.json): these reviewed episodes
        # start with the target resting ON the listed asset.  Declared
        # non-required so the currently-accepted flat pool keeps rendering;
        # flip to required=True per item once its per-episode support
        # placement is calibrated.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="Book", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "3b26eea8-ad70-4b9b-a100-efb049b7bfce",
                    "48127cde-57ac-4e1a-9563-e62da9f155cd",
                    "d6d9d7bb-a71e-4102-826e-f044adbc57ba",
                )),
            dict(
                prim="Note", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "86b4b301-7fea-4fac-878f-3311e076101b",
                    "e5645202-2fc2-4f1f-9e7b-0bea12314b44",
                )),
            dict(
                prim="CushionOnBook", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "3f4f2e42-4ed5-4775-b99d-4179139eddcc",
                    "44c3f5b4-58c5-432f-acc2-9fe7a2dd810f",
                    "6222933c-9d2e-4b49-83a0-4d84ff35991a",
                    "63e00b55-ff4a-4262-bc56-6b3ef6da7c4a",
                    "736469b7-59c6-45f3-a730-33dec78a44ca",
                    "8096385a-7c58-45e0-a9cc-dc82277899b4",
                    "aca722c4-3e0a-4514-9afa-8ad65501cd08",
                    "bc72046f-a8a4-4fe5-b126-438b806f9306",
                )),
            dict(
                prim="PaperCup", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "82bf02e7-0c62-477e-a30f-96f1cc127edf",
                )),
        ),
    ),
    "55": dict(
        family="pick", key="55", root=ASSETS, prim="Wallet",
        zh="find钱包@茶几", new=(4, 8),
        # 2026-07-24 head-video audit: the three shortest complete recordings
        # all show a clean right-hand edge/corner lift of the canonical soft
        # herringbone pouch.  Pinning them avoids manifest-order drift in
        # unattended contact-v1 canaries.
        preferred_probes=(
            "7c7a3f0d-d368-4286-9274-e4f8041c9859",
            "f60be28b-c295-4a31-9f07-cc57f4270739",
            "6fa97ae0-cf4e-49ca-9fe0-1712e0d83d89",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "56": dict(
        family="pick", key="56", root=ASSETS, prim="Remote",
        zh="find遥控器@CoffeeTable（保留机器人模板，不再绑定4.14）",
        new=None,
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="Note", relation="supports_target",
                required=True, kinematic=True,
                episode_uids=(
                    "003d1033-5f4f-4030-9484-0f2aa53495e6",
                )),
            dict(
                prim="Wallet", relation="supports_target",
                required=True, kinematic=True,
                episode_uids=(
                    "07ec98c4-9e86-4abe-a301-43ec193068e6",
                )),
            dict(
                prim="Book", relation="supports_target",
                required=True, kinematic=True,
                episode_uids=(
                    "0a5c5dba-3cf7-4f9f-9d18-8117670e37cd",
                )),
        ),
        preferred_probes=(
            "003d1033-5f4f-4030-9484-0f2aa53495e6",
            "07ec98c4-9e86-4abe-a301-43ec193068e6",
            "0a5c5dba-3cf7-4f9f-9d18-8117670e37cd",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "57": dict(
        family="pick", key="57", root=ASSETS, prim="GlassesCase",
        zh="find墨镜盒@茶几", new=(4, 2),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31 (see
        # episode_support_context_audit_v1.json): these reviewed episodes
        # start with the target resting ON the listed asset.  Declared
        # non-required so the currently-accepted flat pool keeps rendering;
        # flip to required=True per item once its per-episode support
        # placement is calibrated.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="Note", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "1a3fe2f5-212f-4fc8-ac6e-6d4740719d73",
                    "1ff49aeb-34cb-4b09-b3ab-485fa9469534",
                    "30b1f5e6-4945-49ed-a7da-2c5613999ed8",
                    "8b69b95b-a5e0-4ef9-a5a3-c10058e0a0f1",
                    "9c063492-9139-43a6-b69d-5b65e07fa411",
                    "b8bae4c1-67e7-4133-89ed-5e41e34b4560",
                    "e95097bb-892d-4c07-bba4-5136e7a287aa",
                    "ec0b07d4-976f-4a9a-a9ef-386a0e8293a8",
                )),
            dict(
                prim="PaperPlate", relation="supports_target",
                required=False, kinematic=True,
                episode_uids=(
                    "c1621b9b-c54f-4bea-98a2-ec916ff29b1b",
                )),
        ),
    ),
    "58": dict(
        family="pick", key="58", root=ASSETS, prim="Remote",
        # The 0804-v8 benchmark sheet has a remote pick on the snack shelf
        # (4.16) and a remote place onto the TV cabinet (4.17), but no remote
        # pick *from* the TV cabinet.  Do not silently bind this extra robot
        # template to either different atomic action.
        zh="find遥控器@TV柜", new=None,
        preferred_probes=(
            "001648fd-8d51-47cc-ad71-2155d9152e89",
            "08b4636c-e2d8-4c4a-a285-acd244a9fb42",
            "0afea995-452e-4991-9612-811b6ccb1476",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "59": dict(
        family="pick", key="59", root=ASSETS, prim="GlassesCase",
        zh="find墨镜盒@wooden side table", new=(4, 3),
        preferred_probes=(
            "d15fc3f0-595d-4df1-b279-fb0ce29a7832",
            "d6baa895-493c-4e6b-842d-bbcbe697b3db",
            "da0f93de-e7cf-45d7-8e71-d2854de7b79a",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "60": dict(
        family="pick", key="60", root=ASSETS, prim="Wallet",
        zh="find钱包@小茶几", new=(4, 9),
        # Same audit criterion as Task55; these are the three shortest clean
        # left-hand corner/edge lifts on the small table.
        preferred_probes=(
            "9bf60322-9845-4a22-9680-d30bef7caac7",
            "6c629ecf-26a7-4826-9678-b71955ade6d9",
            "b8baa530-8b3e-4452-bb8d-52e03b946c60",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "61": dict(
        family="pick", key="61", root=ASSETS, prim="Remote",
        zh="find遥控器@CoffeeTable（保留为6.9 donor）", new=None,
    ),
    "109": dict(
        family="pick", key="109", root=ASSETS, prim="Remote",
        zh="遥控器@wooden side table（L2 real identity）", new=(4, 14),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "405": dict(
        family="pick", key="405", root=ASSETS, prim="GlassesCase",
        zh="墨镜盒@边柜", new=(4, 3),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "78": dict(
        family="pick", key="78", root=ASSETS, prim="RedBullPink",
        zh="空易拉罐@茶几(pick red bull pink slim)", new=None,
        # 2026-08-04 batch: 60 validated recordings (meta_info task_id 78,
        # working copies under the local record root).  The empty 250 mL
        # pink slim can stands on the CoffeeTable placemat in the shared
        # tidy-table scene (same set as small Tasks 29/30).
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "90": dict(
        family="pick", key="90", root=ASSETS, prim="Umbrella",
        zh="伞@木质边桌", new=(5, 1),
        # 2026-07-31 batch (robot task 90, wooden side table in front of the
        # fireplace — same support as Task 59).  The three shortest recordings
        # whose final head frame shows the folded umbrella lifted clear.
        preferred_probes=(
            "cd515215-6734-464e-baee-9c77cbbdcf8c",
            "ce15a633-1bc1-4316-923e-72fd4da73a3d",
            "c8dc0a33-f768-4f4a-98b8-1b4c756a81e1",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
        # Frame-zero support audit 2026-07-31
        # (episode_support_context_audit_v1.json): d671a31f drapes the
        # umbrella over a lying coke can; every other episode was reviewed
        # flat on the table.  Asset-on-asset initial constructions are
        # allowed; each supported episode needs its own reviewed placement
        # (task90_secondary_context_v1.json) before it renders.
        contact_context_scope="per_episode",
        contact_context=(
            dict(
                prim="ShelfItems/diet_coke", relation="resting_on_target",
                required=True, kinematic=False,
                episode_uids=(
                    "d671a31f-5fa0-4a75-b14e-94da989e1a8a",
                )),
        ),
        contact_context_free_episode_uids=(
            "c50ba26f-f672-4bbd-b057-cf09c76530f1",
            "d3b8108d-a39b-4590-8b9b-1a1a5848ec50",
            "d49993b3-3f4e-45e3-916e-963b426e74fe",
            "5a8cb05a-5b66-4bcf-9127-bb7e6c234dcd",
            "c6b91f9a-076b-4228-880b-a1f9bf99a369",
            "0119c688-ae3f-4512-aeb1-d07c0e4d91ea",
            "58c677d4-b1ad-44a0-a9c2-c92d54fbf1b1",
            "c98f085d-5970-477c-b534-0a8d5941251a",
            "ce15a633-1bc1-4316-923e-72fd4da73a3d",
            "c8dc0a33-f768-4f4a-98b8-1b4c756a81e1",
            "cefac040-9af0-471d-a6ce-c59f9968f92d",
            "cd515215-6734-464e-baee-9c77cbbdcf8c",
        ),
    ),
    "91": dict(
        family="pick", key="91", root=ASSETS, prim="GlassesCase",
        zh="find墨镜盒@沙发座面", new=(4, 1),
        # 2026-07-31 batch (robot task 91) — benchmark 4.1 / Z03 coverage.
        # Three shortest recordings ending with the case held in the gripper.
        preferred_probes=(
            "b5333ead-edb3-40b6-aff1-42f65d896868",
            "b94101a1-8d6a-4f58-9fb0-73d58511bb70",
            "5c8b33d3-4481-4b73-8643-f17f41ad4f3a",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    "92": dict(
        family="pick", key="92", root=ASSETS, prim="Wallet",
        zh="find钱包@茶几抽屉", new=(4, 10),
        # 2026-07-31 batch (robot task 92): the right hand pinches the
        # slightly-open black_box_2 drawer front and pulls it out on the
        # shared articulated-drawer physics (drawer_articulation.py); the
        # left hand lifts the herringbone wallet out.  Frame-0 waist pitch
        # is 45 deg at the front-facing table_pick station.
        preferred_probes=(
            "5e5365d3-e645-4633-a490-03ef5754f665",
            "199da05e-56d9-4881-871f-f489a51e55b0",
            "5db629eb-c8dd-4b3a-a27a-ac24b3550ef5",
        ),
        required_evidence_contract="real2sim.pick_contact.v1",
    ),
    # --- place: the green Sales Order/bill is carried into CoffeeTable drawer 1 ---
    "28": dict(
        family="place",
        key="28",
        root=ASSETS,
        prim="Note",
        zh="账单→茶几抽屉",
        new=(6, 5),
    ),
    # --- throw/trash: pick from CoffeeTable, carry, then release into a bin
    # task 29 was frame-audited on 3b4487d2: three striped PaperCups go one-by-one
    # to the default teal trash bin.  The current playback can track
    # only one PaperCup, so it is explicitly a single-cup partial probe; it must
    # never publish full-task success.  In the output taxonomy this is the
    # placement half of tidy_table's disposable-cup cleanup.
    "29": dict(family="throw", key="29", root=ASSETS, prim="PaperCup",
               zh="3个条纹纸杯→普通垃圾桶（teal）", new=(7, 10),
               required_count=3, probe_scope="single_cup_partial",
               benchmark_segment_contracts={
                   "7.10": "complete_pick_handoff_throw_e2e_episode",
               },
               preferred_probes=(
                   "3b4487d2-fabc-4ebd-83ee-408f00145e38",
               ),
               required_evidence_contract="real2sim.throw_metrics.v1"),
    # 30 = robot "trash paper".
    "30": dict(
        family="throw", key="30", root=ASSETS, prim="PaperBall",
        zh="纸团→桶(trash paper)", new=None,
        benchmark_segment_contracts={
            "7.2": "complete_pick_throw_e2e_episode",
        },
    ),
    # --- cloth: spread on table (Newton pipeline — not init-dispatched here) ---
    "52": dict(family="cloth", key="52", root=ASSETS, prim="Tablecloth", zh="铺桌布", new=(8, 10)),
}

    # One robot small task may supply a reviewed benchmark identity in addition
    # to ``new``. Family 7 uses complete E2E disposal episodes as of 2026-08-08;
    # its former pick/throw segment aliases are intentionally absent.
BENCHMARK_SUBTASK_ALIASES = {
    "11": ("6.4",),
    "12": ("6.1",),
    "13": ("6.1",),
    "29": ("7.10",),
    "30": ("7.2",),
    "31": ("6.8",),
    "37": ("3.25",),
    "41": ("3.26",),
    "53": ("5.9", "6.14"),
    "55": ("7.13",),
    "57": ("7.15",),
    "71": ("5.2",),
    "73": ("5.12",),
    "74": ("5.10", "6.16", "7.14"),
    "66": ("2.3",),
    "67": ("2.4",),
    "91": ("6.17",),
    "109": ("7.17",),
}


def benchmark_subtasks(cid):
    """Return benchmark tasks that may use one legacy adapter/reference."""
    task_id = str(cid)
    entry = REGISTRY.get(task_id, {})
    identities = set(BENCHMARK_SUBTASK_ALIASES.get(task_id, ()))
    primary = entry.get("new")
    if primary is not None:
        identities.add(f"{int(primary[0])}.{int(primary[1])}")
    benchmark_only = entry.get("benchmark_subtask")
    if benchmark_only is not None:
        identities.add(str(benchmark_only))
    return tuple(
        sorted(
            identities,
            key=lambda value: tuple(int(part) for part in value.split(".")),
        )
    )


# Make the policy state visible to reporting/UI callers without deleting
# blocked historical task identities from the canonical registry.
for _cid, _entry in REGISTRY.items():
    _asset = task_asset_decision(_cid)
    if _entry.get("root") in (SHELF, TIER2) and not _asset.applicable:
        raise RuntimeError(
            f"retail task {_cid} is missing from shelf_asset_policy.TASK_ASSETS"
        )
    if _asset.applicable:
        _entry["asset_id"] = _asset.asset_id
        _entry["asset_status"] = _asset.status
        _entry["asset_reason"] = _asset.reason


def distract_prims(cid, resolved_target=None):
    """Legacy retail inventory; init_task replaces it with interactive props."""
    e = REGISTRY.get(cid, {})
    fam, root = e.get("family"), e.get("root", ASSETS)
    tgt = resolved_target or e.get("prim")
    if fam == "snack":
        names = [n for n in _SNACK_DIST if n != tgt]
    elif fam == "tier2":
        names = [n for n in _TIER2_GRID if n != tgt]
    else:
        return []
    return [f"{root}/{n}" for n in names]


def track_prim(cid, resolved_target=None):
    """full USD path of the tracked target prim (resolved_target overrides for runtime picks)."""
    e = REGISTRY[cid]
    return f"{e['root']}/{resolved_target or e.get('prim', '')}"


def episode_contact_context(cid, episode_uid):
    """Return required context bound to one exact reviewed episode UUID."""
    entry = REGISTRY.get(str(cid), {})
    context = tuple(entry.get("contact_context", ()))
    if not context:
        return ()
    if entry.get("contact_context_scope") != "per_episode":
        raise ValueError(
            f"task {cid} contact_context must declare per_episode scope")
    uid = str(episode_uid)
    selected = []
    for item in context:
        episode_uids = tuple(
            str(value) for value in item.get("episode_uids", ()))
        if item.get("required") is True and not episode_uids:
            raise ValueError(
                f"task {cid} required contact_context item "
                f"{item.get('prim')} has no reviewed episode_uids")
        if uid in episode_uids:
            selected.append(item)
    if any(item.get("required") is True for item in context) and not any(
            item.get("required") is True for item in selected):
        # Mixed pools: a task may have SOME episodes whose target rests on
        # another asset and others where it lies flat.  Flat episodes stay
        # usable only after an explicit frame-zero review places them on the
        # task's contact_context_free_episode_uids list; anything unreviewed
        # still fails closed.
        context_free = tuple(
            str(value)
            for value in entry.get("contact_context_free_episode_uids", ()))
        if uid in context_free:
            return ()
        raise ValueError(
            f"task {cid} has no reviewed contact_context for episode {uid}")
    return tuple(selected)


def contact_prims(cid, episode_uid=None):
    """Full paths of required close-contact context.

    With ``episode_uid`` this returns only the exact reviewed episode's
    context.  Without it, callers get the stable de-duplicated task inventory.
    """
    entry = REGISTRY.get(cid, {})
    root = entry.get("root", ASSETS)
    context = (
        episode_contact_context(cid, episode_uid)
        if episode_uid is not None
        else tuple(entry.get("contact_context", ()))
    )
    paths = []
    for item in context:
        path = f"{root}/{item['prim']}"
        if item.get("required") is True and path not in paths:
            paths.append(path)
    return paths


def requires_episode_contact_context(cid):
    """Whether init must resolve an exact reviewed episode context record."""
    entry = REGISTRY.get(str(cid), {})
    context = tuple(entry.get("contact_context", ()))
    if not context:
        return False
    if entry.get("contact_context_scope") != "per_episode":
        raise ValueError(
            f"task {cid} contact_context must declare per_episode scope")
    for item in context:
        if item.get("required") is True and not tuple(
                item.get("episode_uids", ())):
            raise ValueError(
                f"task {cid} required contact_context item "
                f"{item.get('prim')} has no reviewed episode_uids")
    return any(item.get("required") is True for item in context)
