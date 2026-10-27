"""Bounded CPU checks; these do not establish physics or policy success."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from .situations import PROFILE, SCENARIOS, VARIANTS
from .templates import configuration, select, templates


def config(profile=PROFILE):
    return configuration(root="/unused/bank", source_root="/unused/bank/source",
                         sdk_site="/unused/sdk", node="/unused/node", context_profile=profile)


class SituationTests(unittest.TestCase):
    def test_legacy_and_long_tasks_unchanged(self):
        baseline = templates()
        for name, row in baseline.items():
            self.assertEqual(select(config(None), name, 1), row)
            if name not in SCENARIOS:
                self.assertEqual(select(config(), name, 1), row)

    def test_120_deterministic_requests_and_public_boundary(self):
        c = config()
        original = deepcopy(c)
        for scenario in SCENARIOS:
            counts = Counter()
            for seed in range(1, 31):
                row = select(c, scenario, seed)
                self.assertEqual(row, select(c, scenario, seed))
                self.assertEqual(row["request"]["seed"], seed)
                self.assertEqual(row["request"]["layout_id"], c["templates"][scenario]["request"]["layout_id"])
                self.assertEqual(set(row["public_input"]), {"prompt", "context"})
                text = row["public_input"]["context"]
                self.assertTrue(text.startswith(c["templates"][scenario]["public_input"]["context"].rstrip()))
                self.assertEqual(text.count("Today's situation"), 1)
                self.assertNotIn("task_parameters", text)
                self.assertNotIn("umbrella_transfer", text)
                self.assertNotIn("fixed_cpu_bag", text)
                counts[row["context_variant"]["id"]] += 1
            self.assertEqual(counts, {v["id"]: 5 for v in VARIANTS[scenario]})
        self.assertEqual(c, original)

    def test_weather_not_keyword_trigger(self):
        for seed in range(1, 31):
            row = select(config(), "prepare_to_leave", seed)
            v = VARIANTS["prepare_to_leave"][(seed - 1) % 6]
            umbrella = [t for t in row["request"]["tasks"] if t["type"] == "umbrella_transfer"]
            self.assertEqual(bool(umbrella), bool(v.get("umbrella")))
            if umbrella:
                self.assertEqual(umbrella, [dict(id="umbrella", type="umbrella_transfer", source="sofa")])
            if v["id"] in ("forecast_updated_dry", "rain_after_return"):
                self.assertIn("rain", row["public_input"]["context"])
                self.assertFalse(umbrella)

    def test_visible_state_is_not_disclosed_in_daily_context(self):
        # General household rules may mention cleaning conditionally. Today's
        # facts must not identify current objects, waste, or stain presence.
        for variants in VARIANTS.values():
            for variant in variants:
                self.assertNotRegex(variant['today'].lower(),
                    r'\b(stains?|spills?|spilled|cans?|cups?|paper|wrapping|empty|finished|tabletop)\b')
        self.assertEqual(len({v['today'] for v in VARIANTS['tidy_room']}), 1)
        self.assertEqual(select(config(), 'prepare_dinner', 1)['public_input'],
                         select(config(), 'prepare_dinner', 4)['public_input'])
        self.assertEqual(select(config(), 'host_guests', 3)['public_input'],
                         select(config(), 'host_guests', 6)['public_input'])

    def test_preferences_updates_and_prerequisites(self):
        # Revised visitor gets only the confirmed person's products.
        guest = select(config(), "host_guests", 4)
        self.assertEqual({t["product"] for t in guest["request"]["tasks"] if "product" in t}, {"pring_bbq", "coke_c3"})
        pair = select(config(), "host_guests", 5)
        self.assertEqual(len([t for t in pair["request"]["tasks"] if "product" in t]), 4)
        ryan = select(config(), "prepare_dinner", 3)
        self.assertTrue({"coffee", "coffee_to_meal_tray"} <= {t["type"] for t in ryan["request"]["tasks"]})
        cancelled = select(config(), "prepare_dinner", 6)
        self.assertFalse({"coffee", "coffee_to_meal_tray"} & {t["type"] for t in cancelled["request"]["tasks"]})
        for scenario in SCENARIOS:
            for seed in range(1, 31):
                row = select(config(), scenario, seed)
                variant = VARIANTS[scenario][(seed-1) % 6]
                self.assertEqual(any(t["type"] == "wipe_table" for t in row["request"]["tasks"]), bool(variant.get("wipe")))

    def test_existing_inventory_contract_all_120(self):
        # Real original CPU request validation: no mock scorer or SDK imports.
        room = Path(__file__).resolve().parents[2] / "backends/profiles/room"
        old_path = sys.path[:]
        sys.path[:0] = [str(room), str(room / "openend/tasks/scene_requirements")]
        try:
            from planner_v2 import compile_request
            for scenario in SCENARIOS:
                for seed in range(1, 31):
                    with self.subTest(scenario=scenario, seed=seed):
                        row = select(config(), scenario, seed)
                        request = deepcopy(row["request"])
                        plan = compile_request(request)
                        self.assertEqual(request, row["request"])
                        self.assertEqual(plan["schema"], "openend.scene_requirements.plan.v2")
                        self.assertEqual(plan["station_request"]["seed"], seed)
        finally:
            sys.path[:] = old_path

    def test_umbrella_existing_score_definition(self):
        room = Path(__file__).resolve().parents[2] / "backends/profiles/room"
        row = json.loads((room / "openend/catalog.json").read_text())["tasks"]["umbrella_transfer"]
        self.assertEqual(row["component_phases"], ["13-1", "14-1"])
        self.assertEqual(row["maximum_score"], 9)
        self.assertEqual(row["maximum_with_contact_credit"], 10)
        self.assertEqual(row["destination"], "fixed_cpu_bag")

    def test_bad_profile_or_seed_rejected(self):
        with self.assertRaises(ValueError):
            configuration(root="x", source_root="y", sdk_site="z", node="n", context_profile="unknown")
        for seed in (0, 31, True, "1"):
            with self.assertRaises(ValueError):
                select(config(), "prepare_to_leave", seed)


if __name__ == "__main__":
    unittest.main()
