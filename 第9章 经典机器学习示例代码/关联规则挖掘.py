from itertools import combinations
from collections import defaultdict


def simple_apriori(transactions, min_support):
    # 统计项集出现次数
    itemsets = defaultdict(int)
    for transaction in transactions:
        for size in range(1, len(transaction) + 1):
            for itemset in combinations(transaction, size):
                itemsets[frozenset(itemset)] += 1

    # 筛选频繁项集
    n_transactions = len(transactions)
    return {k: v / n_transactions for k, v in itemsets.items()
            if v / n_transactions >= min_support}


def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules


# 使用示例
transactions = [
    ["牛奶", "面包", "啤酒"],
    ["牛奶", "尿布"],
    ["牛奶", "面包", "尿布", "啤酒"],
    ["牛奶", "面包", "尿布"],
    ["牛奶", "尿布"]
]

# 挖掘频繁项集（最小支持度40%）
frequent_itemsets = simple_apriori(transactions, min_support=0.4)
print("频繁项集:")
for itemset, support in frequent_itemsets.items():
    print(f"{set(itemset)}: {support:.2f}")

# 生成关联规则（最小置信度70%）
rules = generate_rules(frequent_itemsets, min_confidence=0.7)
print("\n关联规则:")
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} => {set(consequent)} (置信度: {confidence:.2f})")

# 生成示例规则（实际使用前面代码生成的rules）
demo_rules = [
    (frozenset({'牛奶'}), frozenset({'尿布'}), 0.75),
    (frozenset({'面包'}), frozenset({'牛奶'}), 1.0),
    (frozenset({'啤酒'}), frozenset({'牛奶'}), 1.0),
    (frozenset({'尿布', '面包'}), frozenset({'牛奶'}), 1.0)
]
import matplotlib.pyplot as plt
import networkx as nx

# 设置中文字体（使用系统默认黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 可用 'Microsoft YaHei' 或 'KaiTi' 替代
plt.rcParams['axes.unicode_minus'] = False


def show_rules(rules):
    """极简版可视化"""
    plt.figure(figsize=(10, 6))

    # 1. 创建图形
    G = nx.DiGraph()
    for ante, cons, conf in rules:
        G.add_edge("→".join(ante), "→".join(cons), weight=conf)

    # 2. 绘制图形
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos,
            with_labels=True,
            node_size=800,
            node_color='#A0CBE2',
            width=[d['weight'] * 2 for (_, _, d) in G.edges(data=True)],
            edge_color='#444444',
            arrowsize=15,
            font_size=9)

    # 3. 添加标题
    plt.title("关联规则可视化", pad=20)
    plt.box(False)  # 去掉边框
    plt.show()


# 使用示例
test_rules = [
    (frozenset({'牛奶'}), frozenset({'面包'}), 0.8),
    (frozenset({'尿布'}), frozenset({'啤酒'}), 0.6),
    (frozenset({'鸡蛋'}), frozenset({'牛奶'}), 0.7)
]

show_rules(test_rules)