"""V3: expanded original-product catalog, preserving v1/v2 question banks.

Private feature checks verify the authored answer key against the full catalog;
they do NOT prove natural-language uniqueness or visibility in a rendered scene.
No old task, asset, physics, pose, or scoring implementation is modified.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import random
import re

HERE = Path(__file__).resolve().parent
OUT = HERE / 'question_bank_v3'

# Features transcribed from inspected asset textures, not inferred from directory names.
# The two supplemental public manufacturer links are in asset_evidence/manifest.json.
PRODUCTS = {
 'cayman_jacked': ('drink', 'Cayman Jacked Margarita', '含酒精玛格丽塔', 'alcohol lime agave'),
 'gatorade_m': ('drink', 'Gatorade Zero Orange', '无糖橙味运动饮料', 'sports zero_sugar orange electrolytes bottle'),
 'gatorade_t': ('drink', 'Gatorade Zero Glacier Freeze', '无糖冰川口味运动饮料', 'sports zero_sugar glacier electrolytes bottle'),
 'starbucks': ('drink', 'Starbucks Espresso and Cream Light', '浓缩咖啡奶油饮料', 'coffee caffeinated dairy still'),
 'good2grow': ('drink', 'Good2Grow Strawberry Kiwi', '草莓猕猴桃果汁饮料', 'juice_drink strawberry_kiwi spout still bottle no_added_sugar contains_sugar'),
 'grapefruit_can': ('drink', 'Good & Gather Ruby Red Grapefruit Sparkling Water', '无糖西柚气泡水', 'water sparkling unsweetened zero_sugar grapefruit'),
 'coke_c3': ('drink', 'Diet Coke', '健怡可乐', 'cola caffeinated sparkling zero_sugar sweetened'),
 'sprite_c1': ('drink', 'Sprite', '柠檬青柠汽水', 'lemon_lime sparkling caffeine_free contains_sugar sweetened'),
 'chomps_r': ('snack', 'Chomps Original Beef Sticks', '原味牛肉棒', 'beef savory stick mild'),
 'pring_bbq': ('snack', 'Pringles BBQ', '烧烤味薯片', 'crisps bbq crunchy savory'),
 'oreo_gf': ('snack', 'Oreo Double Stuf Gluten Free', '无麸质双倍夹心饼干', 'sandwich_cookie chocolate creme gluten_free sweet'),
 'honeymaid': ('snack', "Honey Maid S'mores Handi-Snacks", '棉花糖巧克力风味蘸酱饼干', 'dip_kit graham smores sweet'),
 'zbar_t': ('snack', "Zbar Protein Cookies 'n Creme", '饼干奶油风味燕麦蛋白棒', 'bar oats protein_label cookies_creme gluten_free sweet'),
 'skit_littles': ('snack', 'Skittles Littles', '迷你彩色糖果', 'candy fruit_flavour tiny multicolour sweet'),
 'red_bull_pink': ('drink', 'Red Bull Pink Edition', '莓果味红牛能量饮料', 'energy_drink caffeinated berries contains_sugar'),
 'lindt': ('snack', 'Lindt Lindor Milk Chocolate', '瑞士莲牛奶巧克力球', 'truffle milk_chocolate chocolate smooth_filling individually_wrapped sweet'),
 'pringles_green': ('snack', 'Pringles Sour Cream & Onion', '酸奶油洋葱味薯片', 'crisps sour_cream onion crunchy savory'),
}

# Each row: situation/need, evaluator-only necessary features, explanation.
# The question contains neither the asset identifier nor the product/brand name.
CARDS = {
 'cayman_jacked': [
  ("The adult host is off duty and explicitly wants an alcoholic lime drink for a toast, not a soft-drink substitute.", "alcohol lime", "成人明确要含酒精的青柠庆祝饮品，而非看到绿色罐子就选。"),
  ("An adult guest wants a ready-mixed bar-style drink with agave and lime. There is no time to make one from separate ingredients.", "alcohol agave lime", "酒吧式青柠龙舌兰需求对应预调玛格丽塔。"),
  ("The adults have finished dinner and requested the alcoholic option; the citrus sodas are being saved for people who do not drink alcohol.", "alcohol", "区分含酒精饮品和外观相似的汽水。"),
  ("For an adult's cocktail hamper, the item must already contain alcohol and have lime flavour. A mixer alone would leave the hamper incomplete.", "alcohol lime", "完成鸡尾酒礼篮需要成品含酒精饮料，而非调酒用汽水。"),
  ("An adult visitor liked the lime-and-agave drink at yesterday's bar. They want that style again, including the alcohol, not merely a lime-flavoured refreshment.", "alcohol lime agave", "回忆的是口味及含酒精类型，不是品牌名。"),
 ],
 'gatorade_m': [
  ("After practice, I want an electrolyte drink without sugar. I prefer the flavour of the citrus fruit normally squeezed into orange breakfast juice.", "electrolytes zero_sugar orange", "运动补电解质需求加无糖、橙味偏好，排除蓝色款与果汁。"),
  ("The runner wants a zero-sugar sports drink and dislikes the icy-flavour version. Their usual choice tastes of oranges.", "sports zero_sugar orange", "先按运动饮料筛选，再排除另一口味。"),
  ("My training partner asked for electrolytes, not a stimulant. They chose an orange flavour and do not want sugar in today's drink.", "electrolytes orange zero_sugar", "训练需求不是提神需求，结合口味与零糖约束。"),
  ("For the workout bag, pack the sugar-free sports drink matching the orange flavour we agreed on, not something selected just because its bottle looks orange.", "sports zero_sugar orange", "功能、零糖与口味共同约束，不凭包装颜色。"),
  ("The athlete's note says: replace electrolytes; skip sugar; use the citrus flavour, not the one named after ice. Find the drink that meets all three.", "electrolytes zero_sugar orange", "组合三条要求，冰川款不能满足口味条件。"),
 ],
 'gatorade_t': [
  ("After practice, I want a sugar-free electrolyte drink. The orange flavour belongs to my teammate; I want the other sports-drink flavour.", "electrolytes zero_sugar glacier", "运动补液需求加橙味排除，选另一个运动饮料。"),
  ("The runner's usual zero-sugar recovery drink has an ice-themed flavour name. A fruit juice or a fizzy drink will not match that routine.", "sports zero_sugar glacier", "由习惯、零糖和冰雪命名线索匹配。"),
  ("For my next training session I need electrolytes, but I am tired of the orange version. Keep the sugar-free type and change to the icy-flavour option.", "electrolytes zero_sugar glacier", "保留功能和零糖约束，只更换口味。"),
  ("The gym order is for the sports drink whose flavour evokes a frozen landscape. Do not substitute a blue snack package or a zero-sugar soda.", "sports glacier", "功能类别和冰川语义结合，不直接报品牌或只匹配颜色。"),
  ("My teammate likes the citrus sports drink; I prefer the zero-sugar version named for ice. Choose for me, not for my teammate.", "sports zero_sugar glacier", "将偏好绑定到请求者，再选择对应口味。"),
 ],
 'starbucks': [
  ("I have a thesis deadline tonight and want something caffeinated to help me stay alert. I dislike fizz and prefer a creamy drink.", "caffeinated still dairy", "熬夜提神需求，加无气泡和奶油口感，排除含咖啡因的可乐。"),
  ("The early-shift worker wants a ready-to-drink pick-me-up with milk or cream. They are not asking for a sugary citrus soda.", "caffeinated dairy", "清晨提神并要奶味，对应含咖啡因的奶咖。"),
  ("There is no kettle in the editing room. I still want my usual roasted-bean pick-me-up with a creamy finish, ready to drink straight away.", "coffee dairy", "从烘焙豆、提神和无冲泡设备推断即饮奶咖。"),
  ("I am proofreading late and want a stimulant in my drink, but carbonation bothers my taste preference. Milk and cream are welcome.", "caffeinated still dairy", "咖啡因与无气泡共同筛选，奶油偏好进一步限定。"),
  ("For the overnight design session, my favourite flavour is roasted beans softened with cream. I want the ready-made drink, not equipment or a snack.", "coffee dairy", "由口味和使用情境推断咖啡饮料。"),
 ],
 'good2grow': [
  ("My little cousin keeps tipping open cups over. Their parent requested a fruit drink with a spill-resistant drinking spout for the drawing table.", "juice_drink spout", "根据使用困难推断需要带防泼洒吸嘴的果汁饮料。"),
  ("I am packing a fruit drink for a bumpy stroller trip. A drinking spout is more useful here than an open can, and the child likes strawberry with kiwi.", "juice_drink spout strawberry_kiwi", "出行情境、吸嘴和复合水果口味共同限定。"),
  ("The child likes the pairing of a red berry and the green fruit with black seeds. Their parent asked for that flavour in a drink, not candy.", "juice_drink strawberry_kiwi", "由水果描述推断草莓猕猴桃，排除水果糖。"),
  ("We have no spare cups at the picnic. For the child's fruit drink, choose the bottle designed for drinking through a spill-resistant spout.", "juice_drink spout", "缺杯子情境推断包装功能，而非点名商品。"),
  ("For the painting corner, the parent prefers a fruit drink in a reusable bottle with a drinking spout. An ordinary sports-drink cap is not the requested feature.", "juice_drink spout bottle", "儿童活动场景中区分吸嘴与普通瓶盖。"),
 ],
 'grapefruit_can': [
  ("I want bubbles with dinner, but no sweet taste from sugar or sweeteners. A citrus-scented water would fit better than a diet soft drink.", "water sparkling unsweetened", "无甜味不同于零糖，排除甜味的健怡汽水和运动饮料。"),
  ("My friend likes the aroma of the large citrus fruit with pink flesh. They want it in unsweetened fizzy water, not juice or a cocktail.", "grapefruit sparkling unsweetened", "水果常识加饮品类型约束，推断西柚气泡水。"),
  ("I already have a sweet dessert. For the drink, keep the fizz but remove the sweetness entirely; fruit aroma is fine.", "sparkling unsweetened water", "餐搭偏好要求有气无甜，不能把零糖甜汽水当答案。"),
  ("The guest enjoys citrus and carbonation but rejects both sugary and artificially sweet-tasting soft drinks. Choose the plain-water style of refreshment.", "water sparkling unsweetened", "双重排除糖与甜味汽水，保留果味水。"),
  ("At the tasting table, I need a fizzy drink that will not add sweetness between bites. The pink-fleshed citrus aroma is the one I like.", "grapefruit sparkling unsweetened", "试吃情境与西柚偏好共同限定无甜气泡水。"),
 ],
 'coke_c3': [
  ("I have late work to finish and want a caffeinated drink with bubbles. It must have no sugar; I still enjoy a sweet taste.", "caffeinated sparkling zero_sugar sweetened", "熬夜但要气泡与零糖，不能选奶咖或无甜气泡水。"),
  ("I like the dark caramel-style soda taste, but I am choosing the zero-sugar option today. A citrus water would change the flavour too much.", "cola zero_sugar", "保留可乐口味同时要求零糖，匹配实际健怡贴图。"),
  ("The editor wants a fizzy pick-me-up without sugar, not the creamy non-fizzy option. Artificial sweetness is acceptable to them.", "caffeinated sparkling zero_sugar", "以气泡区分两种提神饮料，以零糖排除含糖汽水。"),
  ("For my late-night game, I want the familiar dark soda flavour and caffeine, but none of the sugar of a regular soft drink.", "cola caffeinated zero_sugar", "夜间活动加口味、咖啡因及零糖复合需求。"),
  ("I enjoy sweet-tasting cola, but my current preference is zero sugar. Do not confuse that with water that has no sweet taste at all.", "cola sweetened zero_sugar", "区分甜味、糖分和商品类型。"),
 ],
 'sprite_c1': [
  ("I am winding down for bed and do not want caffeine. I still want a sweet, fizzy lemon-and-lime drink, not plain flavoured water.", "caffeine_free sweetened sparkling lemon_lime", "睡前避免咖啡因，加甜味柠檬青柠汽水偏好。"),
  ("The guest likes the flavours of both yellow lemons and green limes. They want a sweet soda without alcohol, not an unsweetened citrus water.", "lemon_lime sweetened sparkling", "水果组合和甜汽水要求排除青柠酒与西柚水。"),
  ("For a non-alcoholic party drink, I want the lemon-and-lime bubbles with sweetness. The lime cocktail and the unsweetened water are not suitable substitutes.", "lemon_lime sparkling sweetened", "派对需求先排除酒，再按甜味和口味选择。"),
  ("My friend avoids evening caffeine but does not mind sugar. Their favourite drink is a sweet citrus soda, rather than a milk-based pick-me-up.", "caffeine_free lemon_lime sweetened", "不摄入咖啡因不等于无糖，组合偏好匹配。"),
  ("I have already bought the unsweetened fizzy water. Now I need the sweet lemon-and-lime soft drink for the guest who dislikes bitter citrus flavours.", "lemon_lime sweetened sparkling", "区分已购买物和另一位客人的口味需求。"),
 ],
 'chomps_r': [
  ("My hiking partner prefers a savoury, chewy meat snack. They do not want a sweet oat bar even if that also advertises protein.", "beef savory stick", "户外点心需求加肉类咸口，排除蛋白棒。"),
  ("The guest likes beef and wants a ready-to-eat snack with a mild flavour. Dessert biscuits and smoky crisps would not satisfy that preference.", "beef mild", "偏好牛肉及温和口味，不只识别红色包装。"),
  ("For the road trip, I want something savoury I can nibble in stick form. I mean actual meat, not a grain-based bar.", "beef stick savory", "食用方式与原料约束结合。"),
  ("I have enough sweet snacks for the afternoon. My own choice should be the beef-based, mildly seasoned option instead.", "beef mild savory", "已备甜食是背景，个人需求是温和牛肉点心。"),
 ],
 'pring_bbq': [
  ("For movie night, I want a crunchy savoury snack with a smoky grill-like flavour. A chewy meat stick is not what I have in mind.", "crisps bbq crunchy", "电影零食需求，烧烤口味加酥脆质地。"),
  ("My friend likes the taste associated with a barbecue, but wants thin crunchy pieces rather than meat or sweet biscuits.", "bbq crisps crunchy", "烧烤偏好不等于肉类，结合形态排除。"),
  ("The game-night snack should crunch and taste smoky. The sweet dipping kit belongs to dessert, not this request.", "crisps bbq", "区分咸口看球零食与甜味蘸酱套装。"),
  ("We already have chewy snacks. Add the crisp, savoury option with barbecue seasoning so the texture and flavour are different.", "crisps bbq savory", "为已有食物补充不同质地，结合口味限定。"),
 ],
 'oreo_gf': [
  ("I want to twist two dark chocolate biscuit halves apart and eat the pale filling first. A bar that merely has a similar flavour will not work.", "sandwich_cookie chocolate creme", "预期操作需要真实夹心饼干，不是饼干味能量棒。"),
  ("For dessert decoration, I need dark biscuit discs with a white centre that can be split into halves. Candy pieces and oat bars have the wrong shape.", "sandwich_cookie chocolate creme", "装饰用途推断结构与形状，不直接点名。"),
  ("The tea guest prefers cream-filled chocolate sandwich biscuits and asked for a package labelled gluten-free. The oat bar is for somebody else.", "sandwich_cookie chocolate gluten_free", "口味结构与标签偏好共同限定，不作过敏安全承诺。"),
  ("The child wants to imitate the twist-and-dunk biscuit game at the table. Pick the dark sandwich biscuits, not the separate sticks-and-dip kit.", "sandwich_cookie chocolate", "游戏操作与食物结构对应，排除另一蘸食形式。"),
 ],
 'honeymaid': [
  ("We cannot light a campfire indoors, but the guest wants that graham-and-s'mores dessert flavour. They enjoy dipping the sticks themselves.", "dip_kit graham smores", "露营甜点回忆加蘸食方式，对应真实贴图而非花生。"),
  ("For the snack activity, each child should dip a crunchy stick into a separate sweet compartment. A pre-filled sandwich biscuit would skip that activity.", "dip_kit graham", "参与式蘸食需要分离的饼干棒和蘸酱。"),
  ("My friend likes the campfire dessert made with graham crackers and a marshmallow-chocolate flavour. Find the ready-made dipping version, with no cooking needed.", "dip_kit smores graham", "由甜点常识推断风味，再限定免烹饪蘸酱套装。"),
  ("We have plates but no time to melt chocolate. I need the packaged sweet snack that already keeps the graham sticks and their s'mores-style dip together.", "dip_kit graham smores", "缺少制作时间推断现成蘸酱点心。"),
 ],
 'zbar_t': [
  ("For my cycling break, I want an oat-based bar that advertises protein. I like cookies-and-cream flavour, but do not want actual sandwich biscuits.", "bar oats protein_label cookies_creme", "运动间歇需求加燕麦蛋白标签，排除同口味饼干。"),
  ("I want the taste of cookies and cream in a bar-shaped oat snack. A package of loose filled biscuits is the wrong form.", "bar oats cookies_creme", "相近口味需结合食品形态，而非关键词匹配。"),
  ("The commuter asked for a sweet oat bar with a protein claim on the pack, not the savoury meat snack offered by their colleague.", "bar oats protein_label", "蛋白需求存在干扰，用甜燕麦棒条件消歧。"),
  ("Pack the snack for the person who prefers oats and cookies-and-cream flavour in a bar. The person who likes twisting biscuits is not the recipient.", "bar oats cookies_creme", "绑定收件人的习惯，排除夹心饼干。"),
 ],
 'skit_littles': [
  ("For decorating the cupcakes, I need lots of tiny colourful sweet pieces, not large biscuits or bars. A fruit-flavoured candy would fit.", "candy tiny multicolour fruit_flavour", "装饰用途推断尺寸、颜色与糖果形态。"),
  ("The party game needs edible counters in several colours. Choose small fruit-flavoured sweets, not crumbs from a biscuit.", "candy tiny multicolour", "计数游戏功能需要可分拣的小彩色糖果。"),
  ("My guest likes fruity sweets that can be eaten a few tiny pieces at a time. They are not asking for a fruit drink or a chocolate biscuit.", "candy tiny fruit_flavour", "水果偏好需区分糖果、饮料和饼干。"),
  ("We are making a rainbow pattern on dessert plates. Find the miniature multicoloured candies rather than a single large sweet snack.", "candy tiny multicolour", "图案设计需求推断多色小颗粒糖果。"),
 ],
}

CARDS.update({
 'red_bull_pink': [
  ("I have a late editing deadline and want a caffeinated energy drink. I prefer berry flavour and do not mind sugar; a creamy coffee or a cola is not my choice.", "energy_drink caffeinated berries", "熬夜背景结合能量饮料、莓果口味，排除奶咖和可乐。"),
  ("My friend asked for the berry-flavoured energy drink for their overnight project. The electrolyte drink is for tomorrow's training, not tonight's request.", "energy_drink berries", "区分今晚的提神饮料请求与明天的运动补液。"),
  ("We are sorting supplies by purpose: a sports drink for practice and a caffeinated berry energy drink for the late work session. Choose for the work session.", "energy_drink caffeinated berries", "先解析当前活动，再结合莓果口味消歧。"),
  ("The night-shift colleague rejected coffee and cola. Their updated request is a berry-flavoured drink specifically labelled as an energy drink; sugar is acceptable.", "energy_drink berries contains_sugar", "根据更新后的功能与口味选择，不只根据咖啡因。"),
 ],
 'lindt': [
  ("For a small thank-you gift, I want individually wrapped milk-chocolate treats with a smooth centre, not chocolate biscuits or fruity candy.", "truffle milk_chocolate smooth_filling", "礼物用途结合牛奶巧克力和柔滑夹心，排除饼干与水果糖。"),
  ("My guest likes chocolate that melts around a soft filling. They want round chocolate treats, not two biscuit halves with cream between them.", "truffle chocolate smooth_filling", "根据食物结构区分巧克力球与夹心饼干。"),
  ("We already have crunchy snacks. Add the individually wrapped milk-chocolate option with a smooth filling for people who prefer a softer sweet.", "individually_wrapped milk_chocolate smooth_filling", "根据已有零食补充不同质地的甜食。"),
  ("The host wants to put a wrapped chocolate treat at each place setting. Choose milk-chocolate truffles, not a package of loose colourful sweets.", "truffle milk_chocolate individually_wrapped", "桌面布置用途需要独立包装的牛奶巧克力球。"),
 ],
 'pringles_green': [
  ("For movie night, I want crunchy savoury crisps with a tangy cream-and-onion flavour. The smoky barbecue version is for someone else.", "crisps sour_cream onion", "同为薯片，按酸奶油洋葱偏好排除烧烤味。"),
  ("My friend changed their mind about the smoky grill seasoning. Keep the crunchy potato snack, but choose the sour-cream-and-onion flavour instead.", "crisps sour_cream onion crunchy", "保留食品类型，按更新后的口味选择。"),
  ("We have enough sweet snacks. The missing savoury choice should be crisp and onion-flavoured with a sour-cream note, not chewy meat.", "crisps onion sour_cream savory", "咸口、酥脆及口味约束共同消歧。"),
  ("One guest prefers barbecue seasoning; I prefer tangy cream with onion on my crisps. Choose for me, not the other guest.", "crisps sour_cream onion", "收件人绑定后按不同薯片口味选择。"),
 ],
})

FAMILIES = (
 'recipient_binding', 'schedule_branch', 'latest_plan', 'remaining_need',
 'deadline_priority', 'role_swap', 'correction_scope', 'conditional_fallback',
)


def satisfies(product, required):
    return set(required) <= set(PRODUCTS[product][3].split())


def choose_third(a, b):
    pool = [k for k, p in PRODUCTS.items() if p[0] == PRODUCTS[a][0] and k not in (a, b)]
    # Match overlapping properties where possible, without consulting positions.
    features = set(PRODUCTS[a][3].split()) | set(PRODUCTS[b][3].split())
    return min(pool, key=lambda k: (-len(features & set(PRODUCTS[k][3].split())), k))


def pair_text(family, a, b, index):
    """Two questions share context and stock; a changed query reverses the answer."""
    if family == 'recipient_binding':
        context = f"Maya left this note: {a} Leo left this note: {b}"
        scenarios = ("Leo has already been served. Select the item for Maya's outstanding request.",
                     "Maya has already been served. Select the item for Leo's outstanding request.")
    elif family == 'schedule_branch':
        context = f"The participant wrote two plans. For the work session: {a} For the leisure session: {b}"
        scenarios = ("The schedule now puts the work session first. Choose for that session, not the later one.",
                     "The schedule now puts the leisure session first. Choose for that session, not the later one.")
    elif family == 'latest_plan':
        context = f"An earlier request said: {a} A proposed alternative said: {b}"
        scenarios = ("The requester rejected the alternative and reconfirmed the earlier request. Select for the confirmed need.",
                     "The requester accepted the alternative and cancelled the earlier request. Select for the confirmed need.")
    elif family == 'remaining_need':
        context = f"Two people need items. The person at the window says: {a} The person at the door says: {b}"
        scenarios = ("The person at the door has received their item. Fill the other person's request without duplicating the completed one.",
                     "The person at the window has received their item. Fill the other person's request without duplicating the completed one.")
    elif family == 'deadline_priority':
        context = f"Maya's request: {a} Leo's request: {b} Serve the person who leaves sooner; ignore the order in which they spoke."
        scenarios = ("Maya leaves in five minutes; Leo leaves in twenty. Select the item that must be served first.",
                     "Maya leaves in twenty minutes; Leo leaves in five. Select the item that must be served first.")
    elif family == 'role_swap':
        context = f"The coordinator's personal request: {a} The photographer's personal request: {b} These requests stay with the people, not their job titles."
        scenarios = ("They have swapped jobs. Serve the person who is now the photographer, using that person's original preference.",
                     "They have swapped jobs. Serve the person who is now the coordinator, using that person's original preference.")
    elif family == 'correction_scope':
        context = f"My original need was: {a} My colleague's need was: {b}"
        scenarios = ("My colleague asked to receive my original choice instead. I did not change my own request. Select for me.",
                     "I changed my request to match my colleague's original need. Select for me, not my cancelled preference.")
    elif family == 'conditional_fallback':
        context = f"The visitor gave two needs. First option: {a} Backup option: {b} Use the backup only if the first option is reserved."
        scenarios = ("The first option is not reserved. Select the visitor's first choice.",
                     "The first option is reserved for another person. Both remain physically present; select the allowed backup.")
    else:
        raise ValueError(family)
    return context, scenarios


def build_rows():
    rows = []
    for product, (kind, label, zh, features) in PRODUCTS.items():
        pool = [p for p in PRODUCTS if PRODUCTS[p][0] == kind and p != product]
        for n, (need, criteria, reason) in enumerate(CARDS[product]):
            # Hard distractor drawn by attribute overlap, plus one additional product.
            foil = min(pool, key=lambda p: (-len(set(features.split()) & set(PRODUCTS[p][3].split())), p))
            other = choose_third(product, foil)
            rows.append(dict(kind=kind, target=product, target_zh=zh, context='',
                scenario=need + ' Select one shelf item that fits this situation.',
                required_products=sorted((product, foil, other)),
                family='implicit_need_and_constraints', criteria=criteria.split(),
                rationale_zh=reason, pair_id=None, reasoning_steps=[
                    'Infer the requested function or preference from the situation.',
                    'Combine its constraints and ground them in the visible product.']))
    # 44 drink + 32 snack base cases; keep exactly 100 questions per kind.
    for kind, pair_count in [('drink', 28), ('snack', 34)]:
        products = [p for p in PRODUCTS if PRODUCTS[p][0] == kind]
        pairs = list(combinations(products, 2))
        random.Random('context-reasoning-v2:' + kind).shuffle(pairs)
        for i in range(pair_count):
            a, b = pairs[i % len(pairs)]
            if i % 2:
                a, b = b, a
            ca = CARDS[a][i % len(CARDS[a])]
            cb = CARDS[b][(i + 1) % len(CARDS[b])]
            family = FAMILIES[i % len(FAMILIES)]
            context, scenarios = pair_text(family, ca[0], cb[0], i)
            required = sorted((a, b, choose_third(a, b)))
            pair_id = f'{kind}-contrast-{i+1:02d}'
            for product, card, scenario in ((a, ca, scenarios[0]), (b, cb, scenarios[1])):
                rows.append(dict(kind=kind, target=product, target_zh=PRODUCTS[product][2],
                    context=context, scenario=scenario, required_products=required,
                    family=family, criteria=card[1].split(), pair_id=pair_id,
                    rationale_zh='先解析当前收件人、条件或更新范围，再应用需求约束。' + card[2],
                    reasoning_steps=['Resolve the current person, branch or valid request.',
                        'Infer the function/flavour/format needed by that request.',
                        'Use the product appearance and label to reject near-matching distractors.']))
    random.Random('context-reasoning-v2:question-order').shuffle(rows)
    for i, row in enumerate(rows, 1):
        row['id'] = f'R{i:03d}'
    return rows


BRAND_PATTERN = re.compile(r"cayman|gatorade|starbucks|good\s*2\s*grow|good\s*&\s*gather|coca|\bcoke\b|sprite|chomps|pringles|oreo|honey\s*maid|zbar|skittles|\bclif\b", re.I)


def validate(rows):
    assert len(rows) == 200
    assert Counter(r['kind'] for r in rows) == {'drink': 100, 'snack': 100}
    assert len({(r['context'], r['scenario']) for r in rows}) == 200
    pairs = {}
    for row in rows:
        text = row['context'] + ' ' + row['scenario']
        assert not BRAND_PATTERN.search(text), row['id']
        assert not any(key in text for key in PRODUCTS), row['id']
        assert 'peanut' not in text.lower(), 'No verified peanut target in the asset pool'
        matches = [p for p in PRODUCTS if PRODUCTS[p][0] == row['kind'] and satisfies(p, row['criteria'])]
        assert matches == [row['target']], (row['id'], matches)
        assert len(row['required_products']) == len(set(row['required_products'])) == 3
        assert row['target'] in row['required_products']
        if row['pair_id']:
            pairs.setdefault(row['pair_id'], []).append(row)
    assert len(pairs) == 62
    for pair in pairs.values():
        assert len(pair) == 2 and pair[0]['target'] != pair[1]['target']
        assert pair[0]['context'] == pair[1]['context']
        assert pair[0]['required_products'] == pair[1]['required_products']
    return dict(count=200, by_kind=dict(Counter(r['kind'] for r in rows)),
        by_family=dict(Counter(r['family'] for r in rows)),
        by_product=dict(Counter(r['target'] for r in rows)),
        brand_names_in_prompts=0, contrastive_pairs=62,
        authored_feature_keys_unique_across_full_catalog=True,
        semantic_ambiguity_independently_validated=False, rendered_label_readability_verified=False,
        caveats=['76 implicit-need cases plus 124 questions from 62 contrastive pairs.',
            'Context structures repeat. These are not 200 independent reasoning patterns.',
            'Feature-key uniqueness checks are not an independent natural-language review.',
            'Split by scenario/need family and keep both members of each pair in one split.',
            'No allergy safety, medical advice, caffeine dose or guaranteed physiological effect is claimed.',
            'No verified peanut-flavour snack exists in the current pool; those questions are excluded.'])


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    rows = build_rows()
    report = validate(rows)
    # Preserve v1 and refuse to replace an existing v2 output.
    OUT.mkdir(exist_ok=False)
    old = json.loads((HERE / 'question_bank_v1/public_questions.json').read_text())
    public = dict(schema='shelf_context_questions.v3', status='reasoning_draft', language='en',
        shared_instruction=old['shared_instruction'],
        shared_context='Infer needs from the situation and resolve whose current request applies. '
            'Then use the visible product and its label to choose. Personal preferences and schedules are fictional. '
            'All products needed by the question are stocked. Do not infer the answer from shelf position. '
            'Food preferences in these scenarios are not medical or allergy-safety advice.',
        questions=[{k:r[k] for k in ('id','kind','context','scenario')} for r in rows])
    private = dict(schema='shelf_context_answers.v3', visibility='evaluator_only_never_send_to_agent',
        runtime_bindings_verified=False, answers=[
            {k:r[k] for k in ('id','target','target_zh','required_products','family',
                'criteria','rationale_zh','reasoning_steps','pair_id')} for r in rows])
    write_json(OUT / 'public_questions.json', public)
    write_json(OUT / 'answers_private.json', private)
    write_json(OUT / 'product_identities.json', dict(
        evidence='../asset_evidence/manifest.json',
        products=[dict(id=k,kind=v[0],label_en=v[1],label_zh=v[2],features=v[3].split()) for k,v in PRODUCTS.items()]))
    lines = ['# Shelf reasoning bank v3 — 200 questions', '',
        '100 drinks / 100 snacks. No brand names in public prompts.',
        '76 implicit-needs cases; 62 contrastive pairs (124 questions). Answers below are evaluator-only.',
        'Old v1/v2 are preserved. Full-scene visual ambiguity and model performance remain untested.', '']
    for r in rows:
        lines += [f"## {r['id']} · {r['kind']} · {r['family']}", '', r['context'], r['scenario'], '',
                  f"Answer: {r['target']} — {r['target_zh']}", r['rationale_zh'], '']
    (OUT / 'question_bank_review.md').write_text('\n'.join(lines) + '\n')
    report['evidence_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((HERE / 'asset_evidence').glob('*.png'))}
    report['sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()}
    write_json(OUT / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
