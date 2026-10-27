"""Export authoring inputs and a readable preview; never a ready-state archive."""
import argparse
import json
from pathlib import Path

from .situations import PROFILE, SCENARIOS, VARIANTS
from .templates import select, templates


def export(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh preview directory")
    config = dict(templates=templates(), context_profile=PROFILE)
    cases = [dict(scenario=name, seed=seed, **select(config, name, seed))
             for name in SCENARIOS for seed in range(1, 31)]
    document = dict(schema="robotagentbench.openend_authoring_inputs.v1",
                    profile=PROFILE, status="authoring_inputs_not_saved_initial_states",
                    semantic_situations=24, spatial_seeds_per_situation=5,
                    scene_composition_required=True, instances=cases)
    lines = ["# OpenEnd: daily situations", "",
             "24 semantic situations, each assigned five original spatial seeds. "
             "These are new authoring inputs, not composed/accepted saved states.", "",
             "The original Household Memory is retained verbatim. "
             "The following text is appended as Today's situation. "
             "Confirmed current plans take priority over earlier plans and usual preferences.", ""]
    for scenario in SCENARIOS:
        lines.extend(("## " + scenario.replace("_", " ").title(), ""))
        for i, variant in enumerate(VARIANTS[scenario], 1):
            row = select(config, scenario, i)
            lines.extend(("### " + variant["id"], "",
                          "**Request:** " + row["public_input"]["prompt"], "",
                          "**Today's situation:** " + variant["today"], "",
                          "**Authoring rationale (not a model input):** " + variant["why"], ""))
    output.mkdir(parents=True)
    (output / "authoring_inputs.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "preview.md").write_text("\n".join(lines))
    return dict(status=document["status"], authoring_inputs=len(cases), semantic_situations=24,
                preview=str((output / "preview.md").resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
