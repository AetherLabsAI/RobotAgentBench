"""Public daily context and matching authoring requests, not a new scorer.

This is an opt-in cohort. Never rewrite an existing saved launch or pretend
that changing its public text also changes its physical inventory/score plan.
"""
from copy import deepcopy

PROFILE = "daily_situations_v1"
SCENARIOS = ("prepare_dinner", "host_guests", "prepare_to_leave", "tidy_room")

# Long-term memory stays verbatim. Current facts are a separate public section.
PRIORITY = (
    "The household memory describes usual arrangements. Today's confirmed "
    "plans and exceptions take priority over earlier plans or usual preferences."
)

VARIANTS = {
    "prepare_dinner": (
        dict(id="emma_leftovers", person="Emma", meal="leftover_box",
             today="Emma will be home for dinner after work. She wants her usual evening meal.",
             why="Use Emma's saved-leftovers and fork preference."),
        dict(id="alex_dinner", person="Alex", meal="beef_box",
             today="Alex is coming home for dinner and wants the same arrangement as last time.",
             why="Use Alex's warm beef meal, knife and fork preference."),
        dict(id="ryan_coffee", person="Ryan", meal="beef_box", coffee=True,
             today="Ryan is coming home for dinner and has asked for his usual dinner combination.",
             why="Combine Ryan's reference to Alex's meal with his freshly made coffee preference."),
        dict(id="emma_spill", person="Emma", meal="leftover_box", wipe=True,
             today="Emma will be home for her usual dinner. A drink spilled on the coffee table earlier; the stain has not been cleaned.",
             why="Prepare Emma's dinner and clean the actual stain before the table is used."),
        dict(id="dinner_plan_changed", person="Emma", meal="leftover_box",
             today="The earlier dinner plan was for Alex. That plan is cancelled: Emma is coming home instead and wants her usual evening meal.",
             why="Follow the confirmed person, not the cancelled beef-meal plan."),
        dict(id="ryan_no_coffee", person="Ryan", meal="beef_box",
             today="Ryan will be home for his usual dinner, but he has an early start tomorrow and has asked for no coffee tonight.",
             why="Keep the beef meal and cutlery; today's exception removes coffee preparation and serving."),
    ),
    "host_guests": (
        dict(id="alex_visit", people=("Alex",),
             today="Alex is coming over this afternoon and would like his usual refreshments.",
             why="Recall Alex's Oreos and Sprite pairing."),
        dict(id="emma_visit", people=("Emma",),
             today="Emma is coming over after work and would like the refreshments she enjoyed on her last visit.",
             why="Recall Emma's barbecue Pringles and Coca-Cola pairing."),
        dict(id="ryan_after_training", people=("Ryan",),
             today="Ryan is stopping by straight after training and wants his usual post-training refreshments.",
             why="Use Ryan's post-training beef-stick and orange-sports-drink preference."),
        dict(id="visitor_changed", people=("Emma",),
             today="Alex was expected earlier, but he has cancelled. Emma has confirmed that she is coming instead and would like her usual refreshments.",
             why="Replace Alex's choices with Emma's; do not prepare both cancelled and current requests."),
        dict(id="two_visitors", people=("Alex", "Emma"),
             today="Alex and Emma are both coming over. Each would like their own usual refreshments.",
             why="Combine both remembered pairings: four products on the shared wooden tray."),
        dict(id="ryan_visit_spill", people=("Ryan",), wipe=True,
             today="Ryan is coming straight from training. A drink spilled on the coffee table earlier, and the stain is still there.",
             why="Prepare Ryan's refreshments and clean the actual stain in the serving area."),
    ),
    "prepare_to_leave": (
        dict(id="emma_work_rain", person="Emma", outing="work away from home", umbrella=True,
             today="Emma will walk to the library and work there until late afternoon. The forecast says rain is likely during both walks; her route is outdoors.",
             why="Outdoor travel during rain requires the existing umbrella-to-bag task, in addition to usual packing and tidying."),
        dict(id="emma_work_clear", person="Emma", outing="work away from home",
             today="Emma will work at the library until late afternoon. The forecast is dry for the whole time she will be away.",
             why="Usual work-trip preparation; there is no weather-driven umbrella task."),
        dict(id="alex_shopping_rain", person="Alex", outing="go shopping", umbrella=True,
             today="Alex is walking to the shops. It is raining now and is expected to continue until he returns.",
             why="Combine Alex's shopping routine with protection for the rainy walk."),
        dict(id="ryan_park_showers", person="Ryan", outing="go for a walk in the park", umbrella=True,
             today="Ryan is leaving for a park walk. Showers are forecast while he will be outdoors, and there is no covered route.",
             why="Rain during the outdoor activity adds umbrella packing to the usual park-trip preparation."),
        dict(id="forecast_updated_dry", person="Emma", outing="work away from home",
             today="Emma is working at the library today. An earlier forecast predicted rain, but the latest forecast confirms dry weather throughout her trip; the old forecast is no longer valid.",
             why="Use the latest forecast, not the superseded rain cue."),
        dict(id="rain_after_return", person="Alex", outing="go shopping",
             today="Alex is walking to the shops and will be back by noon. The forecast is dry until evening; rain is expected only after 8 p.m.",
             why="Relate the rain forecast to the trip's time window; rain after the return does not create a packing task."),
    ),
    "tidy_room": (
        dict(id="usual_tidy", today="The shared living area needs its usual tidy-up. There was no spill on the coffee table.",
             why="Return belongings and dispose of actual used packaging; do not invent a wiping task."),
        dict(id="spill_cleanup", wipe=True,
             today="The shared living area needs tidying after a drink was spilled on the coffee table. The stain is still present.",
             why="Combine routine tidying with brush cleaning and return."),
        dict(id="used_cans", waste=("can", "can"),
             today="The drinks in the cans left on the coffee table have been finished. There was no spill.",
             why="Dispose of the actual empty cans in the recycling bin, not the ordinary bin."),
        dict(id="paper_and_cup", waste=("paper", "cup"),
             today="The paper on the coffee table is discarded wrapping, not a note to keep. The paper cup has been used and is no longer needed. There was no spill.",
             why="Use the objects' current use status and the household waste rules."),
        dict(id="mixed_waste", waste=("can", "cup"),
             today="Both drinks left on the coffee table have been finished. The empty can and used paper cup can be cleared away. The tabletop has no stain.",
             why="Send two used containers to their different bins; no wiping is required."),
        dict(id="mixed_waste_spill", waste=("can", "cup"), wipe=True,
             today="Both drinks left on the coffee table have been finished. A spill beside the empty can and used paper cup has left a stain.",
             why="Sort both waste items and clean the actual stain, preserving earlier placements."),
    ),
}

REFRESHMENTS = {
    "Alex": ("oreo_gf", "sprite_c1"),
    "Emma": ("pring_bbq", "coke_c3"),
    "Ryan": ("chomps_r", "gatorade_m"),
}


def build(scenario, seed, baseline):
    """Six authored situations x five original spatial seeds per scenario.

    The seed still goes unchanged to the original physical scene generator.
    No pose, lighting, dynamics, scoring, or admission threshold is overridden.
    """
    if scenario not in SCENARIOS or type(seed) is not int or not 1 <= seed <= 30:
        raise ValueError("Known OpenEnd scenario and an integer seed in 1..30 required")
    variant = VARIANTS[scenario][(seed - 1) % 6]
    result = deepcopy(baseline)
    request = result["request"]
    request["seed"] = seed
    tasks = request["tasks"]
    if scenario == "prepare_dinner":
        for task in tasks:
            if "payload_variant" in task:
                task["payload_variant"] = variant["meal"]
        if variant["meal"] == "beef_box":
            tasks.append(dict(id="knife", type="cutlery_to_meal_tray", asset="meal_knife"))
        if variant.get("coffee"):
            tasks.extend((dict(id="coffee", type="coffee"),
                          dict(id="coffee_served", type="coffee_to_meal_tray")))
        prompt = f"{variant['person']} is coming home for dinner. Could you get things ready?"
    elif scenario == "host_guests":
        tasks[:] = [t for t in tasks if t["type"] != "snack_drink_to_tray"]
        for person in variant["people"]:
            for kind, product in zip(("snack", "drink"), REFRESHMENTS[person]):
                tasks.append(dict(id=f"{person.lower()}_{kind}", type="snack_drink_to_tray", product=product))
        names = " and ".join(variant["people"])
        prompt = f"{names} {'are' if len(variant['people']) > 1 else 'is'} coming over shortly. Please get refreshments ready and make the place ready for the visit."
    elif scenario == "prepare_to_leave":
        if variant.get("umbrella"):
            tasks.append(dict(id="umbrella", type="umbrella_transfer", source="sofa"))
        prompt = f"{variant['person']} is getting ready to {variant['outing']} today. Could you help with the preparations?"
    else:
        prompt = "Please tidy up the room."
        if variant.get("waste"):
            trash = next(t for t in tasks if t["type"] == "trash_transfer")
            trash.pop("count", None)
            trash["items"] = [dict(kind=kind, count=1) for kind in variant["waste"]]
    if variant.get("wipe"):
        tasks.append(dict(id="wipe", type="wipe_table"))
    memory = baseline["public_input"]["context"].rstrip()
    result["public_input"] = dict(
        context=memory + "\n\nToday's situation\n\n" + PRIORITY + "\n\n" + variant["today"] + "\n",
        prompt=prompt,
    )
    # Operator-only provenance. Never concatenate it with model input.
    result["context_variant"] = dict(profile=PROFILE, id=variant["id"],
                                     rationale=variant["why"])
    return result
