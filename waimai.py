import random

food = ['黄焖鸡米饭', '炸鸡汉堡', '猪脚饭', '披萨', '肥牛饭/卤肉饭', '日式拉面',
        '冒菜/火锅/麻辣烫', '干锅', '螺蛳粉', '鸡公煲', '花甲米线', '刀削面',
        '家常菜', '粥', '凉皮肉夹馍', '羊肉汤', '包子/水饺/抄手/煎饼', '兰州拉面']

food = ['黄焖鸡米饭', '炸鸡汉堡', '猪脚饭', '披萨', '肥牛饭/卤肉饭',
        '冒菜/火锅/麻辣烫', '干锅', '螺蛳粉', '鸡公煲', '花甲米线',
        '家常菜', '粥', '凉皮肉夹馍', '羊肉汤', '包子/水饺/抄手/煎饼']

t = list(range(len(food)))
random.shuffle(t)
print(food[t[0]], food[t[1]], food[t[2]])
