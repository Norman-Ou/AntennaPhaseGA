# 官方的
from typing import List
import random

# 我写的
from genetic_algorithm import GeneticAlgorithm
from utils import get_SLL, is_valid_degree_sequence, calculate_prephase
from utils import af_phase_selection as get_AF
from logger import Logger
from constant import N

def fitness_fc(individual: List[int]) -> float:
    """
    遗传算法适应度函数：
    - 输入为一个染色体 individual（即度数序列 x）
    - 若不是合法图的度数序列，直接惩罚返回 -9999
    - 否则计算方向图 AF → 求最大旁瓣电平 SLL
    - 目标是最小化 SLL，因此返回 -SLL（用于最大化）

    :param individual: 染色体（一个长度为 N 的整数列表）
    :return: 适应度值（负的 SLL），越大越好
    """
    # try:
    # 判定是否为合法图度数序列
    if not is_valid_degree_sequence(individual):
        return -9999

    # 计算方向图 AF 和 SLL
    af = get_AF(individual)
    sll, *_ = get_SLL(af)

    return -sll

def individual_func():
    """
    个体生成函数，用于初始化染色体的每个基因。

    返回:
        int: [0, N] 范围内的随机整数
    """
    return random.randint(0, N)

def save_result(
    x: List[int],
    sll: float,
    generation_count:int,
    logger: Logger,
):
    """
    保存最终结果到日志，包括 P 矩阵、AF、SLL、主瓣/副瓣等信息。
    :param x: 最优染色体
    :param sll: 对应的 SLL（正值）
    :param logger: Logger 实例
    """
    P = calculate_prephase(x)
    af = get_AF(x)
    sll, main_lobe_gain_list, main_lobe_theta_list, \
        sidelobe_gain_list, sidelobe_theta_list = get_SLL(af)

    logger.log(
        x, P, af, sll,
        main_lobe_gain_list, main_lobe_theta_list,
        sidelobe_gain_list, sidelobe_theta_list,
        generation_count
    )

if __name__ == "__main__":
    from functools import partial
    logger = Logger("prephase_output") # 传入保存结果的文件夹

    ga = GeneticAlgorithm(
        fitness_func=fitness_fc,
        individual_func=individual_func,
        log_func=partial(save_result, logger=logger),
        population_size=50,
        chromosome_length=N,
        generations=100,
        mutation_rate=0.05,
        crossover_rate=0.9,
        elitism=True
    )
    
    best_x, best_neg_sll = ga.run()

    # 还原真实 SLL（正数）
    best_sll = -best_neg_sll if best_neg_sll != -9999 else float("inf")

    # 打印结果
    print("\nBest x:", best_x)
    print(f"Best SLL (dB): {best_sll:.2f}")
    