"""Explicit table-source wallet/glasses to the dedicated item-storage basket."""
from short_tasks.tasks.pack_object.adapter import (
    request_parts, plan_base, unavailable, bind, normalize, factory_override)

RULE_VERSION = "openend.short_task.table_to_storage_basket.v1"


def resolve(request):
    options, _, _ = request_parts(request, "table_to_storage_basket", RULE_VERSION,
        ("asset", "source", "destination"))
    asset = options.get("asset", "wallet")
    source = options.get("source", "coffee_table.objects")
    destination = options.get("destination", "storage_basket")
    if any(not isinstance(v, str) for v in (asset, source, destination)):
        raise ValueError("Asset/source/destination strings required")
    typ = "glasses_transfer" if asset == "glasses_case" else "wallet_transfer"
    label = "sunglasses case" if asset == "glasses_case" else "wallet"
    plan = plan_base(request, "table_to_storage_basket", RULE_VERSION, typ, asset,
        source, destination, 7, "Pick up the %s from the %s, place it inside the item-storage basket beside the trash bins, and release it." % (
            label, "small side table" if source == "small_table" else "permitted object area on the coffee table"),
        ["Exactly one wallet or glasses case, initially unheld and supported on the selected table.",
         "coffee_table.objects and small_table are distinct explicit source variants.",
         "Destination role is storage_basket, distinct from a cloth basket, trash bin or bag.",
         "Reuse legacy_14_4_origin_zone_plus_actual_basket_mesh_support; real basket support required.",
         "No second pack-object task may claim this target."])
    plan["evidence_policy"]["destination_geometry"] = "legacy_14_4_origin_zone_plus_actual_basket_mesh_support"
    plan["evidence_policy"]["destination_path"] = "/World/Assets/TableclothBasket"
    plan["evidence_policy"]["destination_role"] = "storage_basket"
    plan["evidence_policy"]["legacy_path_is_not_role_identity"] = True
    if asset not in ("wallet", "glasses_case"):
        return unavailable(plan, "unsupported_basket_target", "首版物品收纳篮只支持钱包和墨镜盒。")
    if source not in ("coffee_table.objects", "small_table"):
        return unavailable(plan, "unsupported_table_source", "必须显式选择茶几物品区或小茶几来源。", "load")
    if destination != "storage_basket":
        return unavailable(plan, "wrong_destination_role", "不能用桌布篮、垃圾桶或固定开口包代替物品收纳篮。", "load")
    return plan
