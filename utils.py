import numpy as np
from typing import List, Optional, Tuple
from constant import (
    SCAN_ANGLE_NUM, DIR_ANGLE_NUM, DIR_ANGLES, SCAN_ANGLES,
    thetas, unit_gain_linear,
    N,
    G,
    LAMBDA,
)
from logger import Logger

def af_prephase_matrix(prephase: np.ndarray, final_phase: np.ndarray) -> np.ndarray:
    """
    根据预相位矩阵 prephase 和补偿相位矩阵 final_phase，计算阵列因子 AF。

    参数:
        prephase (np.ndarray): N×N 整数矩阵，0 表示 0°，1 表示 90°
        final_phase (np.ndarray): N×N 浮点矩阵，单位：度（通常为 0 或 180）

    返回:
        af (np.ndarray): [SCAN_ANGLE_NUM, DIR_ANGLE_NUM] 的复数矩阵（AF）
    """
    af = np.zeros((SCAN_ANGLE_NUM, DIR_ANGLE_NUM), dtype=complex)

    # 将 prephase 中的 0/1 映射为 0° 和 90°，并转为弧度
    prephase_deg = np.where(prephase == 1, 90, 0)
    phase_pre_rad = np.radians(prephase_deg)

    # final_phase 本来就是角度，直接转为弧度
    # final_phase = np.where(prephase == 1, 180, 0)
    phase_comp_rad = np.radians(final_phase)

    for i in range(SCAN_ANGLE_NUM):
        scan_angle = 2 * np.pi * i / SCAN_ANGLE_NUM - np.pi
        scan_vector = np.array([np.cos(scan_angle), np.sin(scan_angle)])

        # 获取方向图增益
        scan_angle_deg = np.degrees(scan_angle)
        angle_step = thetas[1] - thetas[0]
        gain_idx = int(round((scan_angle_deg - thetas[0]) / angle_step))
        gain_idx = np.clip(gain_idx, 0, len(thetas) - 1)
        element_gain = unit_gain_linear[gain_idx]

        for j in range(DIR_ANGLE_NUM):
            dir_angle_rad = np.radians(DIR_ANGLES[j])
            steer_vector = np.array([np.cos(dir_angle_rad), np.sin(dir_angle_rad)])

            sum_af = 0  # 单个方向的 AF 叠加项

            for s in range(N):
                for k in range(N):
                    # 空间进展相位
                    progressive_phase = (
                        2 * np.pi * G / LAMBDA *
                        (np.dot(scan_vector, [s, k]) - np.dot(steer_vector, [s, k]))
                    )

                    total_phase = (
                        phase_pre_rad[s, k] +
                        phase_comp_rad[s, k] +
                        progressive_phase
                    )

                    sum_af += np.exp(1j * total_phase)

            af[i, j] = sum_af * element_gain

    return af


def af_phase_selection(x: List[int]) -> np.ndarray:
    """
    根据输入向量 x（开关相位分布）计算阵列因子（AF）矩阵。

    参数:
        x (List[int]): 每列允许 0° 元素的数量（其余为 90° 相移）

    全局变量依赖：
        SCAN_ANGLE_NUM: 扫描角的数量（决定扫描分辨率）
        DIR_ANGLE_NUM: 方向角的数量（目标方向）
        DIR_ANGLES:     所有方向角（角度值）
        thetas:         单元方向图角度列表（角度）
        unit_gain_linear: 单元方向图增益（与 thetas 一一对应）
        G: 阵元间距缩放因子（常为1）
        LAMBDA: 波长
        N: 阵列大小（行/列数）

    返回:
        af (np.ndarray): 大小为 [SCAN_ANGLE_NUM, DIR_ANGLE_NUM] 的复数矩阵，
                         表示在不同扫描方向和目标方向下的阵列因子
    """
    af = np.zeros((SCAN_ANGLE_NUM, DIR_ANGLE_NUM), dtype=complex)

    # 遍历每一个扫描方向（观察方向）
    for i in range(SCAN_ANGLE_NUM):
        # 将扫描角映射到 [-π, π]
        scan_angle_rad = 2 * np.pi * i / SCAN_ANGLE_NUM - np.pi
        scan_vector = np.array([np.cos(scan_angle_rad), np.sin(scan_angle_rad)])

        # 将扫描角转换为角度，查找方向图增益
        scan_angle_deg = np.degrees(scan_angle_rad)
        angle_step = thetas[1] - thetas[0]
        gain_idx = int(round((scan_angle_deg - thetas[0]) / angle_step))
        gain_idx = np.clip(gain_idx, 0, len(thetas) - 1)
        element_gain = unit_gain_linear[gain_idx]

        # 遍历每一个方向角（目标方向）
        for j in range(DIR_ANGLE_NUM):
            dir_angle_rad = DIR_ANGLES[j] * np.pi / 180
            steer_vector = np.array([np.cos(dir_angle_rad), np.sin(dir_angle_rad)])

            af_sum = 0  # 该角度下的总和

            # 遍历阵列的每个单元 [s, k]
            for s in range(N):
                for k in range(N):
                    # 相移 = 扫描方向相位差 - 目标方向相位差
                    phase_diff = (
                        2 * np.pi * G / LAMBDA *
                        (np.dot(scan_vector, [s, k]) - np.dot(steer_vector, [s, k]))
                    )

                    # 如果该单元处于“关闭状态”（即 k >= x[k]），添加 π/2 相移
                    if k >= x[s]:
                        phase_diff += np.pi / 2

                    af_sum += np.exp(1j * phase_diff)

            af[i, j] = af_sum * element_gain

    return af


def get_SLL(af: np.ndarray) -> Tuple[float, List[float], List[int], List[float], List[int]]:
    """
    计算方向图 AF 的最大旁瓣电平（SLL）及主瓣/副瓣增益和对应角度

    :param af: 大小为 [扫描角, 方向角数] 的阵列因子复数矩阵
    :return: 
        - sll: 最大旁瓣电平（单位：dB）
        - main_lobe_gain_list: 每个方向上的主瓣增益（dB）
        - main_lobe_theta_list: 每个方向上的主瓣角度索引
        - sidelobe_gain_list: 每个方向上的副瓣最大增益（dB）
        - sidelobe_theta_list: 每个方向上的副瓣角度索引
    """
    def find_local_maxima(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        找到数组中的所有局部极大值及其索引
        :param arr: 输入的一维数组
        :return: (局部极大值数组, 对应索引数组)
        """
        local_max_indices = np.where(
            (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])
        )[0] + 1

        if len(local_max_indices) == 0:
            return np.array([]), np.array([])

        return arr[local_max_indices], local_max_indices
    
    slls = []
    dir_angles_index = [dir_angle + 180 for dir_angle in DIR_ANGLES]

    main_lobe_gain_list = []
    main_lobe_theta_list = []
    sidelobe_gain_list = []
    sidelobe_theta_list = []

    for a, dir_idx in enumerate(dir_angles_index):
        af_temp = af[:, a]  # 第 a 个方向的 AF 切片
        true_dir_angle = DIR_ANGLES[a]

        # 主瓣角度范围（±10度）
        main_lobe_start = dir_idx - 10
        main_lobe_end = dir_idx + 10

        # 找局部极大值
        local_max_values, local_max_indices = find_local_maxima(np.abs(af_temp))

        if len(local_max_indices) == 0:
            print(f"[警告] 方向 {true_dir_angle}° 未找到局部极大值，跳过")
            continue
        
        # import pdb; pdb.set_trace()
        # 主瓣索引（落在主瓣角度范围内的第一个极大值）
        main_lobe_candidates = [
            idx for idx in local_max_indices
            if main_lobe_start <= idx <= main_lobe_end
        ]

        if not main_lobe_candidates:
            print(f"[警告] 方向 {true_dir_angle}° 范围内没有主瓣，跳过")
            continue

        main_lobe_theta = main_lobe_candidates[0]
        main_lobe_gain = af_temp[main_lobe_theta]

        # 去掉主瓣后的剩余局部极大值
        remaining_maxima = [
            (val, idx) for val, idx in zip(local_max_values, local_max_indices)
            if idx != main_lobe_theta
        ]

        if not remaining_maxima:
            sidelobe_gain = 1e-10  # 假设极小副瓣，避免除0
            sidelobe_theta = -1
        else:
            # 最大副瓣
            val_arr, idx_arr = zip(*remaining_maxima)
            sidelobe_gain = max(val_arr)
            sidelobe_theta = idx_arr[val_arr.index(sidelobe_gain)]
        
        # 计算 SLL（单位 dB）
        sll = -20 * np.log10(np.abs(sidelobe_gain) / np.abs(main_lobe_gain))
        slls.append(sll)

        # 存储主瓣/副瓣信息（单位 dB）
        main_lobe_gain_list.append(int(10 * np.log10(np.abs(main_lobe_gain))))
        main_lobe_theta_list.append(main_lobe_theta+SCAN_ANGLES[0])
        sidelobe_gain_list.append(int(10 * np.log10(np.abs(sidelobe_gain))))
        sidelobe_theta_list.append(sidelobe_theta+SCAN_ANGLES[0])

    # 返回所有方向中最大的 SLL
    final_sll = np.mean(slls) if slls else float('inf')
    # import pdb; pdb.set_trace()

    return final_sll, main_lobe_gain_list, main_lobe_theta_list, sidelobe_gain_list, sidelobe_theta_list


def is_valid_degree_sequence(x: List[int]) -> bool:
    """
    判断一个整数序列是否是一个合法的图的度数序列（即是否存在一个简单无向图，
    其每个顶点的度数恰好为 x 中的每个元素）

    参数:
        x (List[int]): 一个整数列表，表示每个节点的期望度数（即希望连接的边数）

    返回:
        bool: 如果这个序列是合法的度数序列，返回 True；否则返回 False

    原理:
        使用 Havel–Hakimi 算法进行验证：不断取出最大度数 d，从后续元素中减去1，
        若过程中出现负数或不能满足连接需求，则说明非法。
    """
    sequence = sorted(x, reverse=True)  # 降序排序，不改变原始列表

    while sequence:
        sequence = [d for d in sequence if d > 0]  # 移除所有0
        if not sequence:
            return True

        d = sequence.pop(0)  # 取出最大度数
        if d > len(sequence):
            return False

        # 减少后面 d 个元素的度数
        for i in range(d):
            sequence[i] -= 1
            if sequence[i] < 0:
                return False

        sequence.sort(reverse=True)  # 重新排序

    return True


def calculate_prephase(x: List[int]) -> Optional[np.ndarray]:
    """
    根据输入的度数序列 x 构造一个对称的 0/1 矩阵 P，其中：
    - P[i, j] = 0 表示第 i 个元素与第 j 个元素连接（互为 0°）
    - P[i, j] = 1 表示未连接（互为 90°）
    
    要求最终每行中 0 的数量等于 x[i]，且生成的矩阵对称（P == P.T）

    参数:
        x (List[int]): 度数序列，x[i] 表示希望第 i 行/列中有 x[i] 个 0

    返回:
        Optional[np.ndarray]: 构造好的 N x N 对称矩阵，如果无法构造则返回 None

    注意:
        - 输入 x 应该是一个合法的图度数序列（可用 is_valid_degree_sequence() 检查）
        - 对角线值始终为 1（不考虑自连边）
    """
    N = len(x)
    P = np.ones((N, N), dtype=int)      # 初始化为全1矩阵，表示默认互为90°
    np.fill_diagonal(P, 1)              # 对角线保持为1（节点不与自己相连）
    x_internal = list(x)                # 拷贝一份 x，避免修改原始数据

    for k in range(N):
        degree_k = x_internal[k]
        if degree_k == 0:
            continue  # 当前节点不需要连接任何人，跳过

        # 找出可连接的候选（尚未连接，度数也 > 0）
        candidates = [
            (i, x_internal[i]) for i in range(N)
            if i != k and x_internal[i] > 0 and P[k, i] == 1
        ]

        if len(candidates) < degree_k:
            print(f"Error: Not enough candidates to match x[{k}] = {degree_k}")
            return None

        # 按剩余度数从大到小排序，优先连接度数大的节点
        candidates.sort(key=lambda item: -item[1])
        selected_indices = [i for i, _ in candidates[:degree_k]]

        # 建立连接：置为0，并更新剩余度数
        for i in selected_indices:
            P[k, i] = 0
            P[i, k] = 0
            x_internal[i] -= 1

        x_internal[k] = 0  # 当前节点连接完成

        # 连接后检查是否有度数变为负数，若有说明不合法
        if any(val < 0 for val in x_internal):
            print(f"Error: Negative degree encountered after processing node {k}")
            return None

    return P


