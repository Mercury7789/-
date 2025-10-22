import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap


# 设置中文字体（临时）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 设置随机种子以保证结果可复现
np.random.seed(42)

# --- 1. 问题定义 (6个资产) ---
# 资产数量
n_assets = 6

# 6只实际股票示例 (代码仅为示意，不代表真实投资建议)
# AAPL, MSFT, AMZN, GOOGL, TSLA, NVDA (数据基于历史均值和波动率估算)
# 预期年化收益率向量 (高收益对应高方差)
returns = np.array([0.06, 0.09, 0.12, 0.14, 0.16, 0.18]) 

# 估算的年化协方差矩阵 (对角线为方差，非对角线为协方差)
# 假设所有股票相关性为0.1 (低相关性)
# 方差与收益率平方大致相关 (vol ~ sqrt(return))
volatilities = np.array([0.11, 0.13, 0.17, 0.19, 0.21, 0.23]) # 对应收益率的波动率
correlation_matrix = np.full((n_assets, n_assets), 0.1) # 假设所有股票相关性为0.1
np.fill_diagonal(correlation_matrix, 1.0) # 自相关为1
# 协方差 = 相关系数 * vol1 * vol2
cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

print("资产信息:")
for i in range(n_assets):
    print(f"  股票{i+1}: 预期收益率 {returns[i]:.1%}, 年化方差 {cov_matrix[i,i]:.4f}, 年化标准差 {np.sqrt(cov_matrix[i,i]):.1%}")

# 算法参数
pop_size = 20
num_generations = 10
mutation_rate = 0.3
crossover_rate = 0.8

# --- 2. 工具函数 ---

def generate_random_weights(n, pop_size):
    """生成满足 sum(w)=1 且 w>=0 的随机权重"""
    weights = np.random.dirichlet(np.ones(n), pop_size)
    return weights

def portfolio_return(weights, returns):#
    return weights @ returns

def portfolio_variance(weights, cov_matrix):#目标函数2，用于计算投资组合的方差（风险）
    return weights.T @ cov_matrix @ weights

def evaluate_population(population, returns, cov_matrix):
    """计算种群的目标值：f1 = -收益, f2 = 方差"""
    f1_list = []
    f2_list = []
    for w in population:
        r = portfolio_return(w, returns)
        v = portfolio_variance(w, cov_matrix)
        f1_list.append(-r)  # 最小化 -收益 => 最大化收益
        f2_list.append(v)   # 目标2：方差
    return np.array(f1_list), np.array(f2_list)

def is_dominates(f1_a, f2_a, f1_b, f2_b):
    """判断 a 是否支配 b"""
    return (f1_a <= f1_b and f2_a <= f2_b) and (f1_a < f1_b or f2_a < f2_b)

def non_domination_sort(f1, f2):
    """非支配排序"""
    n = len(f1)
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    ranks = np.zeros(n, dtype=int)
    ranks.fill(-1)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_dominates(f1[i], f2[i], f1[j], f2[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
    
    current_front = []
    for i in range(n):
        if domination_count[i] == 0:
            ranks[i] = 0
            current_front.append(i)
    
    fronts = [current_front]
    front_idx = 0
    
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    ranks[j] = front_idx + 1
                    next_front.append(j)
        fronts.append(next_front)
        front_idx += 1
    
    return ranks, fronts[:-1] # 去掉最后的空列表

def calculate_crowding_distance(f1, f2, front):
    """计算前沿的拥挤度距离"""
    if len(front) == 0:
        return {}
    if len(front) == 1:
        return {front[0]: float('inf')}
    
    distance = {i: 0.0 for i in front}
    n_obj = 2
    f1_vals = [f1[i] for i in front]
    f2_vals = [f2[i] for i in front]
    
    for obj_vals in [f1_vals, f2_vals]:
        sorted_indices = sorted(range(len(obj_vals)), key=lambda x: obj_vals[x])
        # 边界点无穷大
        distance[front[sorted_indices[0]]] = float('inf')
        distance[front[sorted_indices[-1]]] = float('inf')
        
        min_val, max_val = obj_vals[sorted_indices[0]], obj_vals[sorted_indices[-1]]
        if max_val > min_val:
            for j in range(1, len(sorted_indices)-1):
                idx = front[sorted_indices[j]]
                left_val = obj_vals[sorted_indices[j-1]]
                right_val = obj_vals[sorted_indices[j+1]]
                distance[idx] += (right_val - left_val) / (max_val - min_val)
    
    return distance

def tournament_selection(ranks, crowding_distances, k=2):
    """二元锦标赛选择"""
    def select():
        candidates = np.random.choice(len(ranks), k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and crowding_distances.get(c, 0) > crowding_distances.get(best, 0):
                best = c
        return best
    return select

def crossover(parent1, parent2):
    """算术交叉: child = α*parent1 + (1-α)*parent2"""
    alpha = np.random.uniform(0.3, 0.7)
    child = alpha * parent1 + (1 - alpha) * parent2
    return child / np.sum(child) # 归一化

def mutate(weights, rate, strength=0.1):
    """高斯扰动 + 修复"""
    if np.random.rand() < rate:
        noise = np.random.normal(0, strength, size=weights.shape)
        weights = weights + noise
        weights = np.clip(weights, 0, None) # 非负
        weights = weights / np.sum(weights) # 归一化
    return weights

def create_next_generation(population, f1, f2, ranks, crowding_distances):
    """使用选择、交叉、变异生成子代"""
    n = len(population)
    parents_idx = [tournament_selection(ranks, crowding_distances)() for _ in range(n)]
    parents = population[parents_idx]
    
    children = []
    for i in range(0, n, 2):
        p1 = parents[i]
        p2 = parents[i+1] if i+1 < n else parents[0]
        if np.random.rand() < crossover_rate:
            c1_weights = crossover(p1, p2)
            c2_weights = crossover(p2, p1)
        else:
            c1_weights = p1.copy()
            c2_weights = p2.copy()
        c1_weights = mutate(c1_weights, mutation_rate)
        c2_weights = mutate(c2_weights, mutation_rate)
        children.append(c1_weights)
        if len(children) < n:
            children.append(c2_weights)
    return np.array(children)

def calculate_hypervolume(f1, f2, reference_point):
    """计算种群的超体积指标"""
    # 获取非支配前沿
    ranks, fronts = non_domination_sort(f1, f2)
    first_front = fronts[0] if fronts else []
    
    if not first_front:
        return 0.0
    
    # 获取第一前沿的目标值
    f1_front = f1[first_front]
    f2_front = f2[first_front]
    
    # 对前沿点进行排序
    sorted_indices = np.lexsort((f2_front, f1_front))
    f1_sorted = f1_front[sorted_indices]
    f2_sorted = f2_front[sorted_indices]
    
    # 计算超体积（矩形面积累加）
    hv = 0.0
    for i in range(len(f1_sorted)):
        if i == 0:
            width = reference_point[0] - f1_sorted[i]
        else:
            width = f1_sorted[i] - f1_sorted[i-1]
        height = reference_point[1] - f2_sorted[i]
        hv += width * height
    
    return hv

def plot_hv_evolution(history):
    """绘制每一代种群的HV值演化曲线"""
    # 固定参考点：f1_min(收益) = 100%, f2_max(方差) = 0
    reference_point = (-1.0, 0.0)
    
    print(f"超体积计算使用的固定参考点: ({reference_point[0]:.6f}, {reference_point[1]:.6f})")
    print(f"  参考点说明: f1_min(收益) = {-reference_point[0]:.3%}, f2_max(方差) = {reference_point[1]:.6f}")
    
    # 计算每一代的HV值
    hv_values = []
    for gen, (_, f1, f2) in enumerate(history):
        hv = calculate_hypervolume(f1, f2, reference_point)
        hv_values.append(hv)
        print(f"第 {gen} 代 HV 值: {hv:.6f}")
    
    # 绘制HV演化曲线
    plt.figure(figsize=(10, 6))
    generations = list(range(len(history)))
    plt.plot(generations, hv_values, 'b-o', linewidth=2, markersize=6, label='HV值')
    
    plt.xlabel('演化代数')
    plt.ylabel('超体积 (HV)')
    plt.title('NSGA-II求解投资组合问题的超体积演化曲线')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加一些统计信息
    final_hv = hv_values[-1]
    max_hv = np.max(hv_values)
    improvement = ((final_hv - hv_values[0]) / hv_values[0]) * 100 if hv_values[0] > 0 else 0
    
    plt.text(0.02, 0.98, f'最终HV: {final_hv:.4f}\n最大HV: {max_hv:.4f}\n改进: {improvement:+.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return hv_values

# --- 3. 初始化与演化 ---
population = generate_random_weights(n_assets, pop_size)
history = [] # 存储每代的 population, f1, f2

print("\n开始 NSGA-II 演化 (目标2: 方差, 6资产, 低相关性0.1)...")

for gen in range(num_generations + 1):  # 包括第0代
    f1, f2 = evaluate_population(population, returns, cov_matrix)
    history.append((population.copy(), f1.copy(), f2.copy()))
    
    print(f"第 {gen} 代：非支配解 f1={f1.min():.4f}~{f1.max():.4f}, f2(方差)={f2.min():.6f}~{f2.max():.6f}")
    
    if gen < num_generations:
        # 非支配排序
        ranks, fronts = non_domination_sort(f1, f2)
        # 计算拥挤度
        crowd_distances = {}
        for front in fronts:
            cd = calculate_crowding_distance(f1, f2, front)
            crowd_distances.update(cd)
        # 生成子代
        children = create_next_generation(population, f1, f2, ranks, crowd_distances)
        # 合并 & 环境选择
        combined_pop = np.vstack([population, children])
        combined_f1, combined_f2 = evaluate_population(combined_pop, returns, cov_matrix)
        combined_ranks, _ = non_domination_sort(combined_f1, combined_f2)
        # 选择前 pop_size 个
        crowd_combined = calculate_crowding_distance(combined_f1, combined_f2, list(range(len(combined_f1))))
        indices = np.arange(len(combined_f1))
        sorted_indices = sorted(indices, key=lambda i: (combined_ranks[i], -crowd_combined.get(i, 0)))
        selected_indices = sorted_indices[:pop_size]
        population = combined_pop[selected_indices]

# --- 4. 创建彩虹色渐变 colormap 并预定义颜色 ---
# 为11代 (0-10) 创建颜色数组
colors = plt.cm.rainbow(np.linspace(0, 1, num_generations + 1))

# --- 5. 可视化 ---
plt.figure(figsize=(8,10))

ax1=plt.subplot(2, 1, 1)
# 5.1 目标空间：收益 vs 风险(方差) - 使用彩虹色渐变
for gen in range(len(history)):
    _, f1, f2 = history[gen]
    returns_vals = -f1 # 转回正收益
    risk_var = f2 # 方差
    ax1.scatter(risk_var, returns_vals, c=[colors[gen]], label=f'第{gen}代', s=30, alpha=0.7, edgecolors='gray', linewidth=0.2)

ax1.set_xlabel('风险(方差)')
ax1.set_ylabel('预期收益')
ax1.set_title('目标空间')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# 5.2 决策空间：权重分布 (使用雷达图)
generations_to_plot = [0, 3, 6, 9, 10] # 选择几代进行展示 (包含最后一代)
angles = np.linspace(0, 2 * np.pi, n_assets, endpoint=False).tolist()
angles += angles[:1] # 完成一个圆圈

ax2 = plt.subplot(2, 1, 2, polar=True)

# --- 动态计算雷达图的径向轴范围 ---
all_avg_weights_to_plot = []
for gen in generations_to_plot:
    pop, _, _ = history[gen]
    avg_weights = np.mean(pop, axis=0) # 取该代平均权重
    all_avg_weights_to_plot.append(avg_weights)

# 将所有要绘制的平均权重合并为一个数组
all_avg_weights_to_plot = np.array(all_avg_weights_to_plot)
min_weight = all_avg_weights_to_plot.min()
max_weight = all_avg_weights_to_plot.max()

# 为了美观，稍微扩展范围
margin = (max_weight - min_weight) * 0.1 if max_weight > min_weight else 0.05
min_r = max(0, min_weight - margin)
max_r = max_weight + margin

for gen in generations_to_plot:
    # 获取该代的颜色
    color = colors[gen]
    pop, _, _ = history[gen]
    avg_weights = np.mean(pop, axis=0) # 取该代平均权重
    values = avg_weights.tolist()
    values += values[:1] # 完成一个圆圈
    # 绘制时，将原始权重值映射到新的径向范围 [min_r, max_r]
    scaled_values = [min_r + (v - min_weight) * (max_r - min_r) / (max_weight - min_weight) if max_weight > min_weight else (min_r + max_r) / 2 for v in values]
    ax2.plot(angles, scaled_values, 'o-', linewidth=2, color=color, label=f'第{gen}代')
    ax2.fill(angles, scaled_values, alpha=0.25, color=color) # 填充区域，使用相同颜色

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([f'股票{i+1}' for i in range(n_assets)])
ax2.set_ylim(min_r, max_r) # 设置动态范围
ax2.set_theta_zero_location('N')  # 0度在顶部
ax2.set_theta_direction(-1)  # 顺时针
ax2.set_title(f'决策空间\n(各代平均权重分布, 径向轴范围: {min_r:.3f} - {max_r:.3f})', pad=20)
ax2.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0))



plt.tight_layout()
plt.show()

# --- 6. 打印最终结果 ---

for gen in range(len(history)):
    population, f1, f2 = history[gen]
    print(f"\n--- 第 {gen} 代 ---")
    print("组合编号    股票1    股票2    股票3    股票4    股票5    股票6     | 收益      | 方差      | 标准差")
    print("-" * 95)
    for i, w in enumerate(population):
        print(f"组合 {i+1:2d}:   {w[0]:.3f}   {w[1]:.3f}   {w[2]:.3f}   {w[3]:.3f}   {w[4]:.3f}   {w[5]:.3f} "
              f" | {(-f1[i]):.3%} | {f2[i]:.6f} | {(np.sqrt(f2[i])):.3%}")
        
# 验证收敛性：比较各代前沿的范围
print("\n各代目标值范围对比 (方差作为风险):")
print("演化代数      f1收益(Min, Max)        f2风险(Var Min, Max)")
print("-" * 45)
for gen, (_, f1, f2) in enumerate(history):
    print(f"{gen:3d}  ({f1.min():.4f}, {f1.max():.4f})  ({f2.min():.6f}, {f2.max():.6f})")

# 计算并显示平均权重变化
print("\n各代平均权重变化:")
avg_weights_history = []
for gen_data in history:
    avg_weights = np.mean(gen_data[0], axis=0)
    avg_weights_history.append(avg_weights)

avg_weights_history = np.array(avg_weights_history)
for asset_idx in range(n_assets):
    print(f"  股票{asset_idx+1} 平均权重: {avg_weights_history[:, asset_idx].min():.3f} -> {avg_weights_history[:, asset_idx].max():.3f}")

# --- 7. 绘制HV演化曲线 ---
print("\n--- 超体积(HV)演化分析 ---")
hv_evolution = plot_hv_evolution(history)
