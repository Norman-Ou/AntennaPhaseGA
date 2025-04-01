import os
import json
import numpy as np
import matplotlib.pyplot as plt
from constant import DIR_ANGLES

class Logger:
    def __init__(self, base_dir: str):
        """
        初始化 Logger，用于保存遗传算法每代最优解的记录
        :param base_dir: 保存结果的文件夹路径
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def log(
            self, 
            x: list, 
            P: np.ndarray, 
            af: np.ndarray, 
            sll:float,
            main_lobe_gain_list, 
            main_lobe_theta_list, 
            sidelobe_gain_list, 
            sidelobe_theta_list,
            generation_count,
        ):
        """
        保存当前代的 x、P、AF 可视化结果
        :param x: 当前最优个体（整数列表）
        :param P: 对应生成的矩阵（0/1，对称）
        :param af: 计算得到的阵列因子（shape = [360, DIR_ANGLE_NUM]）
        """
        gen_folder = os.path.join(self.base_dir, f"gen{generation_count:03d}_sll{sll:.2f}")
        os.makedirs(gen_folder, exist_ok=True)

        # 保存 x
        result_path = os.path.join(gen_folder, "result.json")
        main_lobe_result = [f"{gain}/{theta}" for gain, theta in zip(main_lobe_gain_list, main_lobe_theta_list)]
        side_lobe_result = [f"{gain}/{theta}" for gain, theta in zip(sidelobe_gain_list, sidelobe_theta_list)]

        result = {
            "x": str(x),
            "sll": sll,
            "main_lobe_gain (db/theta)": main_lobe_result,
            "side_lobe_gain (db/theta)": side_lobe_result,
            "P": str(P.tolist())
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

        # 保存 AF 图像
        af_path = os.path.join(gen_folder, "af.png")
        vis_af(af, save_path=af_path)


def vis_af(af, save_path: str = None):
    af = af.copy()
    dir_angles = DIR_ANGLES

    # 全角度（-180° 到 180°）
    plt.figure(figsize=(12, 6))
    for i in range(af.shape[1]):
        plt.plot(list(range(-180, 180, 1)), 10 * np.log10(np.abs(af[:, i])), label=f"{dir_angles[i]}")
    plt.grid(True)
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

    # 局部角度（-90° 到 90°）
    plt.figure(figsize=(12, 6))
    af = af[90:(360 - 90), :]
    for i in range(af.shape[1]):
        plt.plot(list(range(-90, 90, 1)), 10 * np.log10(np.abs(af[:, i])), label=f"{dir_angles[i]}")
    plt.grid(True)
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path.replace(".png", "_90.png"))
        plt.close()