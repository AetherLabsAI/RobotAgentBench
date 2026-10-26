"""Record the main agent's completed native-image inspection with image hashes."""
import json
from .. import sim2_runner as io
from .report import sha

if __name__=='__main__':
    root=io.HOST/'shelf_continuous_v2';rows=[]
    for key in ('blue904v3','coke907v3'):
        native=root/'native'/key
        rows.append(dict(key=key,reviewed_images={name:sha(native/('0000_'+name+'.png')) for name in ('head','left')},
            manual_visual_review=True,target_visible_supported=True,active_empty_hand_visible=True,
            background_layouts_visibly_different=True,robot_source_poses_unchanged_by_generator=True,
            exact_mesh_gap_inferred_from_picture=False,model_recovery_verified=False))
    io.save(root/'visual_review.json',dict(rows=rows,review_scope='two native heads and two active wrist views',
        source_asset_appearances_preserved=True,global_registration_switched=False))
