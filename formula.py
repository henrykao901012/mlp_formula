import torch
import numpy as np


class PhysicsFormula:
    """基礎物理公式類別"""

    def __init__(self, name, variable_name="x", output_name="y"):
        self.name = name
        self.variable_name = variable_name  # 輸入變數名稱
        self.output_name = output_name  # 輸出變數名稱

    def compute(self, x):
        """計算單一輸入的輸出值"""
        raise NotImplementedError

    def generate_data(self, n_samples, value_range):
        """生成單輸入單輸出的資料"""
        raise NotImplementedError

    def get_description(self):
        """獲取公式描述，包含固定參數"""
        return self.name


class NewtonSecondLaw(PhysicsFormula):
    """F = ma (固定m，輸入a，輸出F)"""

    def __init__(self, m=2.0):
        self.m = m  # 固定質量
        super().__init__(f"F={m}*a", variable_name="a", output_name="F")

    def compute(self, a):
        return self.m * a

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        a = np.random.uniform(min_val, max_val, n_samples)
        f = self.compute(a)

        inputs = a.reshape(-1, 1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"F = ma (m={self.m})"


class KineticEnergy(PhysicsFormula):
    """KE = 0.5 * m * v^2 (固定m，輸入v，輸出KE)"""

    def __init__(self, m=1.0):
        self.m = m  # 固定質量
        super().__init__(f"KE=0.5*{m}*v^2", variable_name="v", output_name="KE")

    def compute(self, v):
        return 0.5 * self.m * v**2

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        v = np.random.uniform(min_val, max_val, n_samples)
        ke = self.compute(v)

        inputs = v.reshape(-1, 1)
        targets = ke.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"KE = 0.5mv² (m={self.m})"


class GravitationalForce(PhysicsFormula):
    """F = G * m1 * m2 / r^2 (固定m1和m2，輸入r，輸出F)"""

    def __init__(self, m1=5.0, m2=10.0):
        self.G = 1.0  # 簡化重力常數
        self.m1 = m1
        self.m2 = m2
        super().__init__(f"F={m1}*{m2}/r^2", variable_name="r", output_name="F")

    def compute(self, r):
        return self.G * self.m1 * self.m2 / (r**2 + 1e-6)

    def generate_data(self, n_samples, value_range=(0.5, 10.0)):
        min_val, max_val = value_range
        # r不能太小，避免數值爆炸
        r = np.random.uniform(max(min_val, 0.5), max_val, n_samples)
        f = self.compute(r)

        inputs = r.reshape(-1, 1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"F = Gm₁m₂/r² (G={self.G}, m₁={self.m1}, m₂={self.m2})"


class IdealGasLaw(PhysicsFormula):
    """PV = nRT => P = nRT/V (固定n和T，輸入V，輸出P)"""

    def __init__(self, n=1.0, T=300.0):
        self.R = 1.0  # 簡化氣體常數
        self.n = n  # 莫耳數
        self.T = T  # 溫度
        super().__init__(f"P={n}*{T}/V", variable_name="V", output_name="P")

    def compute(self, V):
        return self.n * self.R * self.T / (V + 1e-6)

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        V = np.random.uniform(min_val, max_val, n_samples)
        P = self.compute(V)

        inputs = V.reshape(-1, 1)
        targets = P.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"P = nRT/V (n={self.n}, R={self.R}, T={self.T})"


class Pendulum(PhysicsFormula):
    """T = 2π * sqrt(L/g) (輸入L，輸出T)"""

    def __init__(self):
        self.g = 9.8  # 重力加速度
        super().__init__("T=2π*sqrt(L/9.8)", variable_name="L", output_name="T")

    def compute(self, L):
        return 2 * np.pi * np.sqrt(L / self.g)

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        L = np.random.uniform(min_val, max_val, n_samples)
        T = self.compute(L)

        inputs = L.reshape(-1, 1)
        targets = T.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"T = 2π√(L/g) (g={self.g})"


class CoulombsLaw(PhysicsFormula):
    """F = k * q1 * q2 / r^2 (固定q1和q2，輸入r，輸出F)"""

    def __init__(self, q1=2.0, q2=3.0):
        self.k = 1.0  # 簡化庫倫常數
        self.q1 = q1
        self.q2 = q2
        super().__init__(f"F={q1}*{q2}/r^2", variable_name="r", output_name="F")

    def compute(self, r):
        return self.k * self.q1 * self.q2 / (r**2 + 1e-6)

    def generate_data(self, n_samples, value_range=(0.5, 10.0)):
        min_val, max_val = value_range
        r = np.random.uniform(max(min_val, 0.5), max_val, n_samples)
        f = self.compute(r)

        inputs = r.reshape(-1, 1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"F = kq₁q₂/r² (k={self.k}, q₁={self.q1}, q₂={self.q2})"


class HookesLaw(PhysicsFormula):
    """F = kx (輸入x，輸出F)"""

    def __init__(self, k=10.0):
        self.k = k  # 彈簧常數
        super().__init__(f"F={k}*x", variable_name="x", output_name="F")

    def compute(self, x):
        return self.k * x

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        x = np.random.uniform(min_val, max_val, n_samples)
        f = self.compute(x)

        inputs = x.reshape(-1, 1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    def get_description(self):
        return f"F = kx (k={self.k})"
