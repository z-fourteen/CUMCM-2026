"""炉内环境温度模型 Tf(x)

模型三扩展：FurnaceTemperature 类支持可调冷却参数。"""
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


class FurnaceTemperature:
    """可调冷却参数的炉温场模型（模型三）

    保持前面所有温区的 Sigmoid 过渡不变：
        25 -> 175 -> 195 -> 235 -> 255
    只修改最后一个冷却段（255 -> 25）的
    中心位置和过渡宽度。

    参数
    ----
    cooling_center : float
        冷却段 Sigmoid 中点位置 (cm)，默认 342.0
    cooling_width : float
        冷却段 5%-95% 过渡宽度 (cm)，默认 5.0
        当 width=5 时，ac=1.178，与原始 Tf() 完全一致。
        ac = 2*ln(19)/width，保证在 width cm 内完成 5%-95% 温度变化。
    """

    def __init__(self, cooling_center=342.0, cooling_width=5.0):
        self.cooling_center = cooling_center
        self.cooling_width = cooling_width

        # ac = 2*ln(19)/wc 保证在宽度wc内完成5%-95%温度变化
        # wc=5 时 ac≈1.178，与原始一致
        if cooling_width > 0:
            self.ac = 2 * np.log(19) / cooling_width
        else:
            self.ac = 1.178  # fallback

        # 前面温区固定参数
        self.a0 = 1.178

    def temperature(self, x):
        """炉内环境温度 Tf(x) — 可调冷却段"""
        a = self.a0
        # 保持前面温区不变，仅冷却段使用自定义 ac
        return (25
                + 150 * sigma(a * (x - 25))
                + 20 * sigma(a * (x - 200))
                + 40 * sigma(a * (x - 235.5))
                + 20 * sigma(a * (x - 271))
                - 230 * sigma(self.ac * (x - self.cooling_center)))
