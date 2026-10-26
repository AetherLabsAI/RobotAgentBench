"""Product exclusivity by semantic task group, exact ownership by scene instance.

Wood-tray children form ONE group even when each item has its own scorer.
This module neither moves objects nor changes any scoring predicate.
"""
from collections import Counter
from copy import deepcopy
import random
# Read-only product ID/label snapshot of the original fourteen-item catalog.
# Kept dependency-free for the laptop scene-preview planner; parity is tested.
PRODUCT_LABELS=dict(zip(('cayman_jacked','gatorade_m','gatorade_t','starbucks','good2grow',
    'grapefruit_can','coke_c3','sprite_c1','chomps_r','pring_bbq','oreo_gf','honeymaid','zbar_t','skit_littles'),
    ('酒精饮料','橙色运动饮料','蓝色运动饮料','含咖啡因饮料','儿童果汁','气泡水','可乐','雪碧',
     '牛肉棒','薯片','奥利奥饼干','Honey Maid 饼干','蛋白棒','糖果')))

WOOD='snack_drink_to_tray'
MEAL='drink_to_meal_tray'

def validate_groups(tasks):
    claimed={}
    for task in tasks:
        product=task.get('parameters',{}).get('product',task.get('product'))
        if product is None:continue
        if product not in PRODUCT_LABELS:raise ValueError('Unknown product: '+str(product))
        kind=task['type'];identity=task.get('instance_id',task.get('id'))
        group=WOOD if kind==WOOD else (kind,identity)
        if product in claimed and claimed[product]!=group:
            raise ValueError('跨任务品类冲突：'+PRODUCT_LABELS[product]+'不能同时属于木托盘和其他任务，或两个不同任务')
        claimed[product]=group
    return claimed

def expand_scene_tasks(tasks):
    """Quantity -> independent child tasks; same four-item registration limit."""
    validate_groups(tasks)
    result=[];ids=set();original_ids=set();total=0
    for task in tasks:
        if task['id'] in original_ids:raise ValueError('Duplicate original task ID: '+task['id'])
        original_ids.add(task['id'])
        if task['type']!=WOOD:
            rows=[deepcopy(task)]
        else:
            count=task.get('count',1)
            if type(count) is not int or not 1<=count<=4:raise ValueError('货架数量必须是 1–4 的整数')
            total+=count
            if total>4:raise ValueError('饮料和零食合计最多 4 件，含同款多件')
            rows=[]
            for i in range(count):
                row=deepcopy(task);row.pop('count',None)
                row['id']=task['id'] if count==1 else task['id']+'__item_'+str(i+1)
                rows.append(row)
        for row in rows:
            if row['id'] in ids:raise ValueError('Duplicate expanded task ID: '+row['id'])
            ids.add(row['id']);result.append(row)
    return result

def exact_object(parameters,manifest,instance_id):
    product=parameters['product']
    rows=[o for o in manifest['objects'] if o['id']==product and
          o.get('task_role')=='requested_full_product' and o.get('task_id')==instance_id]
    if len(rows)!=1:raise ValueError('Exactly one requested product instance for '+instance_id)
    return rows[0]

def reserve_inventory(plan,requests):
    """Keep sampled layer counts; guarantee requested multiplicities.

    Background has no extra indistinguishable copies of requested products.
    Existing non-requested inventory order remains wherever possible.
    """
    from openend.tasks.scene_requirements.shelf_inventory_policy import POOLS,LABELS
    needed=Counter(r['product'] for r in requests)
    if sum(needed.values())>4:raise ValueError('At most four requested shelf items')
    result=deepcopy(plan)
    for layer in result['layers']:
        sid=layer['support_id'];counts={p:n for p,n in needed.items() if p in POOLS[sid]}
        wanted=sum(counts.values())
        if wanted>layer['count']:raise ValueError('Requested instances exceed sampled shelf capacity')
        names=[o['product_family'] for o in layer['objects']]
        remaining=Counter(counts);free=[]
        for i,name in enumerate(names):
            if name in counts and remaining[name]>0:remaining[name]-=1
            elif name in counts:free.append(i)
        # Reserve missing requests from surplus requested slots, then background.
        free.extend(i for i,n in enumerate(names) if n not in counts)
        for name,n in remaining.items():
            for _ in range(n):names[free.pop(0)]=name
        rng=random.Random(f'product_claims_v1:{plan["seed"]}:{sid}')
        used=Counter();background=[p for p in POOLS[sid] if p not in counts]
        for i,name in enumerate(names):
            if name in counts:
                used[name]+=1
                if used[name]>counts[name]:
                    names[i]=rng.choice(background)
        copies=Counter();objects=[]
        for name in names:
            copies[name]+=1
            objects.append(dict(instance_id=f'shelf/{sid}/{name}/{copies[name]:02d}',
                product_family=name,label=LABELS[name],support_id=sid,
                asset_relative_to_all_assets=f'{name}/asset.usda',required_instance=name in counts))
        if any(copies[p]!=n for p,n in counts.items()):raise ValueError('Inventory multiplicity mismatch')
        layer['objects']=objects;layer['required_counts']=counts
    result['required_counts']=dict(needed)
    result['task_product_background_duplicates']=False
    return result
