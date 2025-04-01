import numpy as np
import pandas as pd

C = 3 * 10 ** 8  # 光速
F = 26.5 * 10 ** 9  # 工作中心频率 26.5GHz
LAMBDA = C / F
G = 5.65 * 10 ** -3  # 单元间隔5.65mm

N = 8

# 扫描角
SCAN_ANGLES = list(range(-180, 180, 1))
SCAN_ANGLE_NUM = len(SCAN_ANGLES)

# 方向角
DIR_ANGLES = list(range(-30, 40, 10)) # [-30, -20, -10, 0, 10, 20, 30]
# DIR_ANGLES = [10]
DIR_ANGLE_NUM = len(DIR_ANGLES)

# 单元
unit_gain_data = pd.read_csv("Unit_Gain_Plot2.csv")
thetas = unit_gain_data["Theta [deg]"].values
unit_gain_db = unit_gain_data["dB(GainTotal) []"].values
unit_gain_linear = 10 ** (unit_gain_db / 10)


if __name__ == "__main__":
    print(SCAN_ANGLES)