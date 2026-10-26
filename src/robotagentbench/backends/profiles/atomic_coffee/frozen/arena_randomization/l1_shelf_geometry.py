"""Canonical world-space geometry for the audited four-tier metal rack."""

from __future__ import annotations


SHELF_TOTAL_HEIGHT_M = 1.5723
SHELF_BOARD_TOP_Z_M = (0.095, 0.751, 1.1825, 1.564)

# Visible boards now match the existing collision planes; keep the legacy
# tier-2 initializer's explicit 0.5 mm settling clearance.
TIER2_OVERLAY_CENTER_Z_M = 0.745
TIER2_ITEM_BASE_Z_M = 0.7515

# Tier-3 drink placement uses the shared visible/collision support surface.
TIER3_BOARD_TOP_Z_M = SHELF_BOARD_TOP_Z_M[2]
TIER3_OVERLAY_CENTER_Z_M = 1.1765

# The source asset is centimetre-authored and the room rotates its local +Y
# into world +Z.  A true 0.01 scale and zero Z translation preserve the model's
# audited metric dimensions in every consuming room layer.
SHELF_SCENE_LOCAL_Y_SCALE = 0.01
SHELF_SCENE_TRANSLATE_Z_M = 0.0
