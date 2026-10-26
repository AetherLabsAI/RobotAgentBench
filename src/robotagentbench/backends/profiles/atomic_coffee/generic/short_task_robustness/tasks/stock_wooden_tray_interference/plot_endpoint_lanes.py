"""Measured shelf-local top view of ordinary vs interference CPU compositions."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Patch
from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'openend/tasks/scene_requirements'),
               str(ROOT/'layout_task_activation_v1/task_scene_registration_v1'),
               str(ROOT/'api-protocol')]
from repair_room_placement_v2 import shape_mesh, cs


def draw():
    base = HERE/'integration/endpoint_lanes_v1'
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, directory, title in zip(axes, ('ordinary_seed8','room_seed8'),
                                   ('Ordinary short task: one access lane',
                                    'Robustness variant: both endpoints reserved')):
        manifest = json.loads((base/directory/'composition.json').read_text())
        stage = Usd.Stage.Open(manifest['scene'])
        cache = UsdGeom.XformCache()
        inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/RoomEpisode/Shelf')).GetInverse()
        init = manifest['shelf_initialization']
        if directory == 'ordinary_seed8':
            lanes = init['placement_policy']['lanes']
        else:
            lanes = manifest['environment_reservations'][0]['endpoint_grasp_lanes']['lanes']
        for i, lane in enumerate(lanes):
            color = '#2787d9' if i == 0 else '#e68b25'
            ax.add_patch(Polygon(lane['polygon_xy_m'],closed=True,
                                 facecolor=color,alpha=.22,edgecolor=color,linewidth=2))
        for row in init['placements_in_shelf_frame']:
            if row['support_id'] != 'tier3': continue
            parts=[]
            for prim in Usd.PrimRange(stage.GetPrimAtPath(row['prim_path'])):
                mesh=shape_mesh(prim)
                if mesh is None: continue
                visible=(UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'
                         and UsdGeom.Imageable(prim).ComputePurpose() in ('default','render'))
                if not visible and not prim.HasAPI(UsdPhysics.CollisionAPI): continue
                mesh.apply_transform(cs.matrix(cache.GetLocalToWorldTransform(prim)*inverse))
                parts.append(np.asarray(mesh.vertices))
            points=np.concatenate(parts);lo,hi=points.min(0),points.max(0)
            target=row['product_family']=='gatorade_t'
            ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],facecolor='#2787d9' if target else '#a4adb7',
                                   edgecolor='#28323c',alpha=.85))
            ax.text((lo[0]+hi[0])/2,hi[1]+.012,'TARGET' if target else row['product_family'],
                    ha='center',fontsize=7,rotation=15 if not target else 0)
        if directory == 'room_seed8':
            x0,y0,x1,y1=lanes[1]['target_bounds_xy_m']
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,
                                   edgecolor='#c36b0c',linestyle='--',linewidth=2))
            ax.annotate('EMPTY DESTINATION',xy=((x0+x1)/2,y1),
                        xytext=((x0+x1)/2+.05,y1+.10),ha='center',fontsize=7,
                        color='#995008',arrowprops=dict(arrowstyle='->',color='#995008'))
        ax.axhline(-.25,color='#444444',linestyle='--',linewidth=1)
        ax.set(xlim=(-.65,.65),ylim=(-.32,.30),aspect='equal',title=title,
               xlabel='Shelf-local X (m)',ylabel='Depth Y (m)')
        ax.text(-.62,-.30,'Robot approach / shelf front',fontsize=8,color='#555555')
        ax.grid(alpha=.15)
    fig.legend(handles=[Patch(color='#2787d9',alpha=.25,label='Source access trapezoid'),
                        Patch(color='#e68b25',alpha=.25,label='Destination access trapezoid'),
                        Patch(color='#a4adb7',label='Background item bounds')],
               loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Same seed 8 and target product | CPU placement geometry, not a physical rollout',fontsize=13)
    fig.savefig(base/'comparison.png',dpi=180)


if __name__=='__main__':draw()
