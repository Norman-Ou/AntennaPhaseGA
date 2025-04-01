# second_phase.py

from typing import List
import random
import numpy as np

# 你自己的模块
from genetic_algorithm import GeneticAlgorithm
from utils import get_SLL, calculate_prephase
from utils import af_prephase_matrix as get_AF
from logger import Logger
from constant import N

prephase = np.array(
    [[0, 0, 0, 0, 1, 0, 0, 0],
     [0, 1, 1, 0, 1, 1, 0, 1],
     [0, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 0, 1, 1, 0, 1],
     [0, 0, 0, 0, 1, 0, 1, 0],
     [0, 1, 0, 0, 0, 1, 0, 1]]
)


def fitness_fc(individual: List[int]) -> float:
    """
    第二阶段遗传算法适应度函数：
    - 输入为四个角落的补偿相位（0 或 180）
    - 固定 prephase，构造 final_phase（完整补偿矩阵）
    - 目标是最小化阵列方向图的 SLL

    :param individual: 长度为4的列表，角落 2×2 区域补偿相位（单位：度）
    :return: -SLL，用于最大化适应度
    """
    # try:
    if not valid_func(individual):
        return -9999

    final_phase = np.zeros((N, N), dtype=float)
    # 设置四个角落区域
    corners = [(0, 0), (0, N - 2), (N - 2, 0), (N - 2, N - 2)]
    for i, (row, col) in enumerate(corners):
        phase = individual[i]
        final_phase[row][col] = phase
        final_phase[row + 1][col] = phase
        final_phase[row][col + 1] = phase
        final_phase[row + 1][col + 1] = phase

    af = get_AF(prephase, final_phase)
    af = np.squeeze(af)
    sll, *_ = get_SLL(af)
    return -sll  # 最大化 -SLL → 最小化 SLL
    # except Exception as e:
    #     print(f"[fitness_corner_phase] Error for individual = {individual} -> {e}")
    #     return -9999

def individual_func():
    """
    初始化角落补偿相位个体，每个为 0 或 180（单位：度）
    """
    return random.choice([0, 180])

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
    final_phase = np.zeros((N, N), dtype=float)
    # 设置四个角落区域
    corners = [(0, 0), (0, N - 2), (N - 2, 0), (N - 2, N - 2)]
    for i, (row, col) in enumerate(corners):
        phase = x[i]
        final_phase[row][col] = phase
        final_phase[row + 1][col] = phase
        final_phase[row][col + 1] = phase
        final_phase[row + 1][col + 1] = phase

    af = get_AF(prephase, final_phase)
    sll, main_lobe_gain_list, main_lobe_theta_list, \
        sidelobe_gain_list, sidelobe_theta_list = get_SLL(af)

    logger.log(
        x, final_phase, af, sll,
        main_lobe_gain_list, main_lobe_theta_list,
        sidelobe_gain_list, sidelobe_theta_list,
        generation_count
    )

def valid_func(seq):
    return True

if __name__ == "__main__":
    from functools import partial
    logger = Logger("1bit_output") # 传入保存结果的文件夹

    ga = GeneticAlgorithm(
        fitness_func=fitness_fc,
        individual_func=individual_func,
        valid_func=valid_func,
        log_func=partial(save_result, logger=logger),
        population_size=50,
        chromosome_length=4,
        generations=200,
        mutation_rate=0.05,
        crossover_rate=0.9,
        elitism=True
    )

    best_corner_phase, best_neg_sll = ga.run()
    best_sll = -best_neg_sll if best_neg_sll != -9999 else float("inf")

    print("\nBest corner phase config:", best_corner_phase)
    print("Best SLL (after 2nd phase) (dB):", best_sll)
