"""Camera-only closeups of the composed episode; never move scene objects."""
import itertools
import math
import numpy as np


def fit(bounds, front=(0,-1), elevation=.7):
    lo,hi=np.asarray(bounds,dtype=float);target=(lo+hi)/2
    front=np.asarray(front,dtype=float)
    norm=np.linalg.norm(front)
    # A vertical local axis has virtually no XY projection. Never place a camera
    # inside its target by treating that projection as a unit horizontal vector.
    front=front/norm if np.isfinite(norm) and norm>1e-8 else np.array([0.,-1.])
    back=np.r_[front*math.sqrt(1-elevation**2),elevation]
    right=np.cross([0,0,1],back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    corners=np.array(list(itertools.product(*zip(lo,hi))))-target
    x,y,z=corners@right,corners@up,corners@back
    distance=max(.45,float(np.max(z+1.14*np.maximum(abs(x)/(36/56),abs(y)/(22.5/56)))))
    return (target+back*distance).tolist(),target.tolist()


def closeups(record,bounds_of,front_of):
    views=[];objects=record['objects'];furniture=record['overview']['furniture']
    byid={o['id']:o for o in objects}
    def path(name):
        return next((f['path'] for f in furniture if f['path'].endswith('/'+name)),None)
    def add(name,label,paths,front=(0,-1),elevation=.7,pad=.04):
        paths=list(dict.fromkeys(p for p in paths if p))
        boxes=[bounds_of(p) for p in paths];boxes=[b for b in boxes if b is not None]
        if not boxes:return
        lo=np.min(np.asarray(boxes)[:,0],axis=0)-pad;hi=np.max(np.asarray(boxes)[:,1],axis=0)+pad
        eye,target=fit([lo,hi],front,elevation)
        views.append(dict(name='area_'+name,label=label,eye=eye,target=target,bounds=[lo.tolist(),hi.tolist()],target_paths=paths))
    def object_path(key):return byid.get(key,{}).get('path')
    def toward(a,b,fallback=(0,-1)):
        aa,bb=bounds_of(a) if a else None,bounds_of(b) if b else None
        return (np.mean(bb,axis=0)-np.mean(aa,axis=0))[:2] if aa is not None and bb is not None else fallback
    table=path('CoffeeTable');sofa=path('Sofa');small=path('SmallCoffeeTable')
    for task in record['tasks']:
        if task['type'] in ('coffee','fryer'):
            add(task['type'],{'coffee':'咖啡操作区','fryer':'炸锅操作区'}[task['type']],task['object_paths'],task['placement']['front'],.65,.09)
    shelf=object_path('snack_shelf')
    if shelf:
        front=front_of(shelf,(0,-1,0))
        stock=[o['path'] for o in objects if o.get('source')=='snack_shelf']
        add('shelf','货架正面',[shelf,*stock],front,.12)
        if stock:add('shelf_stock','货架商品近景',stock,front,.16,.13)
    add('sofa','沙发拿取区',[sofa,*[o['path'] for o in objects if o.get('source')=='sofa']],toward(sofa,table),.65)
    add('small_table','小茶几拿取 / 放书区',[small,*[o['path'] for o in objects if o.get('source')=='small_table']],toward(small,sofa),.8)
    add('table','茶几整体',[table],toward(table,sofa),.8)
    if record.get('wipe_table'):
        add('wipe','茶几 · 刷子与污渍',[object_path('wipe_tool_hand_brush'),object_path('table_stain_grid')],toward(table,sofa),.85,.07)
    add('cloth','茶几 · 桌布区',[object_path('tablecloth_dirty'),*[o['path'] for o in objects if o.get('source')=='coffee_table.cloth']],toward(table,sofa),.85,.07)
    add('tray','茶几 · 木托盘区',[object_path('japanese_tray')],toward(table,sofa),.8,.09)
    tabletop=[o['path'] for o in objects if o.get('source')=='coffee_table.objects']
    if tabletop:add('table_objects','茶几 · 物品 / 垃圾区',tabletop,toward(table,sofa),.8,.12)
    for key,name,label in [('TvTable','tv','电视柜归位区'),('CoatRack','coat','衣架挂帽区'),('Task5PackStation','pack','装包区'),('TrashBinRed','trash','普通垃圾桶'),('TrashBinTeal','recycle','回收桶')]:
        p=path(key) or '/World/Assets/'+key;add(name,label,[p],toward(p,sofa),.5 if key=='CoatRack' else .7)
    drawer=table+'/black_box_1' if table else None
    add('drawer','茶几抽屉外观',[drawer],toward(table,sofa),.25,.10)
    for key,name,label in [('refrigerator','fridge','冰箱操作区'),('microwave','microwave','微波炉操作区')]:
        p=object_path(key)
        # all_assets fridge geometry is Y-up below its converted root: local +Z
        # is the front; local -Y is vertical after the authored conversion.
        axis=(0,0,1) if key=='refrigerator' else (0,-1,0)
        if p:add(name,label,[p],front_of(p,axis),.25)
        else:
            p='/World/Background/Shell/KitchenRefrigerator' if key=='refrigerator' else path('DiningTableMicrowave')
            if p and bounds_of(p) is not None:add(name,label,[p],front_of(p,axis),.3)
    return views
