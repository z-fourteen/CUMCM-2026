"""炉内环境温度模型 Tf(x)"""
import numpy as np


def sigma(z):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-z))


def Tf(x):
    """炉内环境温度场，x单位：cm

    使用Sigmoid函数构造连续空间温度场，
    平滑参数 a=1.178 已固定。
    """
    a = 1.178
    return (25
            + 150 * sigma(a * (x - 25))
            + 20 * sigma(a * (x - 200))
            + 40 * sigma(a * (x - 235.5))
            + 20 * sigma(a * (x - 271))
            - 230 * sigma(a * (x - 342)))
