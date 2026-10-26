"""UI selection of existing Openend task families; no new scoring rules."""
SHELF_PRODUCTS = list(zip(('cayman_jacked','gatorade_m','gatorade_t','starbucks','good2grow','grapefruit_can','coke_c3','sprite_c1','chomps_r','pring_bbq','oreo_gf','honeymaid','zbar_t','skit_littles'),('酒精饮料','橙色运动饮料','蓝色运动饮料','咖啡饮料','儿童果汁','气泡水','可乐','雪碧','牛肉棒','薯片','奥利奥饼干','Honey Maid 饼干','蛋白棒','糖果')))
CHOICES = [
    dict(type='meal_tray_transfer',name='搬运大餐盘到茶几',group='摆盘任务',description='空盘也可搬；持有并抬离原承托2分，茶几指定餐盘区终态3分；拿走撤回终态分'),
    dict(type='coffee_to_meal_tray',name='咖啡摆盘',group='摆盘任务',description='必须同时选择制作咖啡；取杯过程5分，大餐盘终态3分',requires='coffee'),
    dict(type='bread_to_meal_tray',name='面包摆盘',group='摆盘任务',description='必须同时选择炸锅；取面包过程4分，大餐盘终态3分',requires='fryer'),
    dict(type='cutlery_to_meal_tray',name='餐具摆盘',group='摆盘任务',description='抓住1分、离开承托1分、大餐盘终态3分',variants=[['meal_fork','叉子'],['meal_knife','餐刀']],variant_field='asset',variant_label='餐具'),
    dict(type='salad_to_meal_tray',name='沙拉盒摆盘',group='摆盘任务',description='冰箱开门、取沙拉盒、关门过程8分，大餐盘终态3分'),
    dict(type='drink_to_meal_tray',name='饮料摆盘',group='摆盘任务',description='货架拿取过程4分，大餐盘终态3分；与木托盘任务不能选同品类',variants=SHELF_PRODUCTS[:8],variant_field='product',variant_label='饮料'),
    dict(type='microwave_box_to_meal_tray',name='微波炉取盒摆盘',group='设备任务',description='等待冰箱同一盒子入炉交接 → 旋钮完整亮灯周期 → 开门取盒 → 关门 → 大餐盘；交接前不计后续分',requires='fridge_to_microwave',variants=[['beef_box','牛肉盒'],['leftover_box','剩饭盒']]),
    dict(type='snack_drink_to_tray', name='零食架取物放木托盘', group='物品整理', description='同款可多件、逐件计分，合计最多4件；不同任务品类不可重复', variants=SHELF_PRODUCTS, variant_field='product', variant_label='品类'),
    dict(type='coffee', name='制作咖啡', group='设备任务', description='放杯 → 倒豆 → 按钮 → 取杯'),
    dict(type='fryer', name='炸锅蛋糕', group='设备任务', description='放入蛋糕 → 取出炸篮 → 取走蛋糕'),
    dict(type='fridge_to_microwave', name='冰箱取盒放入微波炉', group='设备任务', description='开冰箱 → 取盒 → 关冰箱 → 放进微波炉并关门', variants=[['beef_box','牛肉盒'],['leftover_box','剩饭盒']]),
    dict(type='remote_transfer', name='遥控器归位', group='物品整理', description='遥控器 → 电视柜', sources=['small_table','sofa']),
    dict(type='glasses_transfer', name='墨镜盒归位', group='物品整理', description='墨镜盒 → 固定开口包或垃圾桶旁收纳篮；7分', sources=['small_table','sofa','coffee_table.objects','coffee_table.drawer'],variants=[['fixed_cpu_bag','固定开口包'],['storage_basket','垃圾桶旁收纳篮']],variant_field='destination',variant_label='目的地'),
    dict(type='wallet_transfer', name='钱包归位', group='物品整理', description='钱包 → 固定开口包或垃圾桶旁收纳篮；7分', sources=['small_table','sofa','coffee_table.objects','coffee_table.drawer'],variants=[['fixed_cpu_bag','固定开口包'],['storage_basket','垃圾桶旁收纳篮']],variant_field='destination',variant_label='目的地'),
    dict(type='powerbank_transfer', name='充电宝装包', group='物品整理', description='充电宝 → 固定开口包', sources=['sofa']),
    dict(type='water_bottle_transfer', name='水瓶装包', group='物品整理', description='水瓶 → 固定开口包', sources=['sofa']),
    dict(type='umbrella_transfer', name='雨伞装包', group='物品整理', description='沙发上的折叠雨伞 → 固定开口包；拿取过程5分、放置过程2分、包内终态2分；另沿用接触奖励', sources=['sofa']),
    dict(type='book_transfer', name='书本归位', group='物品整理', description='书本 → 小茶几', sources=['sofa'], count=True),
    dict(type='cap_transfer', name='帽子挂衣架', group='物品整理', description='帽子 → 衣架', sources=['sofa']),
    dict(type='notebook_transfer', name='笔记本收进抽屉', group='物品整理', description='绿色笔记本 → 抽屉并关上', sources=['sofa']),
    dict(type='trash_transfer', name='扔垃圾', group='清理任务', description='随机 2–4 件；按实例绑定正确垃圾桶', sources=['coffee_table.objects'], count=True),
    dict(type='wipe_table', name='擦拭茶几污渍', group='清理任务', description='抓起刷子1分；真实接触并多次擦拭，按污渍减少量0–4分；擦净放回1分，共6分'),
    dict(type='cloth_clear', name='收拾桌布', group='清理任务', description='随机杂物，或指定书本数量；全部移出桌布并放稳', sources=['coffee_table.cloth'], count=True, cloth_book_selection=True),
]
BY_TYPE = {row['type']: row for row in CHOICES}
