import numpy as np
from typing import List, Tuple, Callable

class NSGAII:
    """NSGA-II (Non-dominated Sorting Genetic Algorithm II) 多目标优化算法框架"""
    
    def __init__(self, pop_size: int, num_variables: int, num_objectives: int, 
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1):
        """
        初始化NSGA-II算法
        
        Args:
            pop_size: 种群大小
            num_variables: 决策变量数量
            num_objectives: 目标函数数量
            crossover_prob: 交叉概率 (默认0.9)
            mutation_prob: 变异概率 (默认0.1)
        """
        self.pop_size = pop_size
        self.num_variables = num_variables
        self.num_objectives = num_objectives
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
    def initialize_population(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """初始化种群
        
        Args:
            bounds: 每个决策变量的边界 [(min, max), ...]
            
        Returns:
            随机生成的初始种群
        """
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.num_variables))
    
    def non_dominated_sort(self, population: np.ndarray, objectives: np.ndarray) -> List[List[int]]:
        """非支配排序 - 将种群划分为多个Pareto前沿
        
        Args:
            population: 种群个体
            objectives: 对应的目标函数值
            
        Returns:
            按前沿等级分组的个体索引列表
        """
        fronts = [[]]
        domination_counts = [0] * len(population)  # 被支配次数
        dominated_solutions = [[] for _ in range(len(population))]  # 支配的解列表
        
        # 计算每个解的支配关系
        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue
                # i支配j
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    dominated_solutions[i].append(j)
                # j支配i
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    domination_counts[i] += 1
            
            # 如果没有被任何解支配，属于第一前沿
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        # 构建后续前沿
        current_front = 0
        while current_front < len(fronts):
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def crowding_distance(self, front: List[int], objectives: np.ndarray) -> np.ndarray:
        """计算拥挤度距离 - 衡量解在目标空间的分布密度
        
        Args:
            front: 同一前沿的个体索引
            objectives: 目标函数值
            
        Returns:
            每个个体的拥挤度距离
        """
        distances = np.zeros(len(front))
        if len(front) == 0:
            return distances
        
        # 对每个目标函数分别计算
        for obj_idx in range(self.num_objectives):
            # 按当前目标函数值排序
            sorted_indices = np.argsort(objectives[front, obj_idx])
            # 边界点的拥挤度设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            min_obj = objectives[front[sorted_indices[0]], obj_idx]
            max_obj = objectives[front[sorted_indices[-1]], obj_idx]
            
            # 计算中间点的拥挤度
            if max_obj - min_obj > 0:
                for i in range(1, len(front)-1):
                    distances[sorted_indices[i]] += (
                        objectives[front[sorted_indices[i+1]], obj_idx] - 
                        objectives[front[sorted_indices[i-1]], obj_idx]
                    ) / (max_obj - min_obj)
        
        return distances
    
    def tournament_selection(self, population: np.ndarray, fronts: List[List[int]], 
                           crowding_distances: List[np.ndarray]) -> List[int]:
        """锦标赛选择 - 基于前沿等级和拥挤度选择父代
        
        Args:
            population: 种群
            fronts: 前沿列表
            crowding_distances: 每个前沿的拥挤度距离
            
        Returns:
            被选中的个体索引列表
        """
        selected = []
        for _ in range(self.pop_size):
            # 随机选择两个个体
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            
            # 找到两个个体所在的前沿
            front1 = next(i for i, front in enumerate(fronts) if idx1 in front)
            front2 = next(i for i, front in enumerate(fronts) if idx2 in front)
            
            # 比较前沿等级
            if front1 < front2:
                selected.append(idx1)
            elif front1 > front2:
                selected.append(idx2)
            else:
                # 同一前沿时比较拥挤度距离
                if crowding_distances[front1][fronts[front1].index(idx1)] > crowding_distances[front2][fronts[front2].index(idx2)]:
                    selected.append(idx1)
                else:
                    selected.append(idx2)
        
        return selected
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉 - 生成子代
        
        Args:
            parent1: 父代个体1
            parent2: 父代个体2
            
        Returns:
            两个子代个体
        """
        if np.random.random() < self.crossover_prob:
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """多项式变异 - 对个体进行变异操作
        
        Args:
            individual: 待变异个体
            bounds: 变量边界
            
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < self.mutation_prob:
                mutated[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return mutated
    
    def evolve(self, population: np.ndarray, objectives: np.ndarray, 
               bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """进化一代 - 执行选择、交叉、变异和环境选择
        
        Args:
            population: 当前种群
            objectives: 当前目标函数值
            bounds: 变量边界
            
        Returns:
            新一代种群和目标函数值
        """
        # 非支配排序和拥挤度计算
        fronts = self.non_dominated_sort(population, objectives)
        
        crowding_distances = []
        for front in fronts:
            crowding_distances.append(self.crowding_distance(front, objectives))
        
        # 锦标赛选择父代
        selected_indices = self.tournament_selection(population, fronts, crowding_distances)
        
        # 生成子代
        offspring = []
        for i in range(0, len(selected_indices), 2):
            if i + 1 < len(selected_indices):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[i + 1]]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                offspring.extend([child1, child2])
        
        # 合并父代和子代
        offspring_population = np.array(offspring[:self.pop_size])
        combined_population = np.vstack([population, offspring_population])
        
        combined_objectives = np.vstack([objectives, 
                                       np.array([self.evaluate_fitness(ind) for ind in offspring_population])])
        
        # 环境选择 - 选择下一代种群
        fronts = self.non_dominated_sort(combined_population, combined_objectives)
        
        new_population = []
        new_objectives = []
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                new_population.extend(combined_population[front])
                new_objectives.extend(combined_objectives[front])
            else:
                # 使用拥挤度距离选择前沿中的个体
                front_crowding = self.crowding_distance(front, combined_objectives)
                sorted_indices = np.argsort(-front_crowding)
                remaining = self.pop_size - len(new_population)
                new_population.extend(combined_population[front][sorted_indices[:remaining]])
                new_objectives.extend(combined_objectives[front][sorted_indices[:remaining]])
                break
        
        return np.array(new_population), np.array(new_objectives)
    
    def evaluate_fitness(self, individual: np.ndarray) -> np.ndarray:
        """评估个体适应度 - 需要用户根据具体问题实现
        
        Args:
            individual: 待评估的个体
            
        Returns:
            目标函数值数组 (最小化问题)
            
        Note:
            用户需要继承此类并实现此方法
        """
        raise NotImplementedError("用户需要实现此方法以定义具体问题的目标函数")
    
    def run(self, bounds: List[Tuple[float, float]], generations: int) -> Tuple[np.ndarray, np.ndarray]:
        """运行NSGA-II算法
        
        Args:
            bounds: 变量边界列表
            generations: 进化代数
            
        Returns:
            最终种群和对应的目标函数值
        """
        # 初始化种群
        population = self.initialize_population(bounds)
        objectives = np.array([self.evaluate_fitness(ind) for ind in population])
        
        # 进化循环
        for gen in range(generations):
            population, objectives = self.evolve(population, objectives, bounds)
        
        return population, objectives
