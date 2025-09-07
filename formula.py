import torch
import numpy as np


class PhysicsFormula:
    """基礎物理公式類別"""

    def __init__(self, name):
        self.name = name

    def compute(self, *args):
        raise NotImplementedError

    def generate_data(self, n_samples, value_range):
        raise NotImplementedError


class NewtonSecondLaw(PhysicsFormula):
    """F = ma"""

    def __init__(self):
        super().__init__("F=ma")

    def compute(self, m, a):
        return m * a

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        m = np.random.uniform(min_val, max_val, n_samples)
        a = np.random.uniform(min_val, max_val, n_samples)
        f = self.compute(m, a)

        inputs = np.stack([m, a], axis=1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class KineticEnergy(PhysicsFormula):
    """KE = 0.5 * m * v^2 (動能公式)"""

    def __init__(self):
        super().__init__("KE=0.5*m*v^2")

    def compute(self, m, v):
        return 0.5 * m * v**2

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        m = np.random.uniform(min_val, max_val, n_samples)
        v = np.random.uniform(min_val, max_val, n_samples)
        ke = self.compute(m, v)

        inputs = np.stack([m, v], axis=1)
        targets = ke.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class GravitationalForce(PhysicsFormula):
    """F = G * m1 * m2 / r^2 (萬有引力，簡化版 G=1)"""

    def __init__(self):
        super().__init__("F=m1*m2/r^2")
        self.G = 1.0  # 簡化重力常數

    def compute(self, m1, m2, r):
        return self.G * m1 * m2 / (r**2 + 1e-6)  # 加小值避免除零

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        m1 = np.random.uniform(min_val, max_val, n_samples)
        m2 = np.random.uniform(min_val, max_val, n_samples)
        r = np.random.uniform(max(min_val, 0.5), max_val, n_samples)  # r不能太小
        f = self.compute(m1, m2, r)

        inputs = np.stack([m1, m2, r], axis=1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class IdealGasLaw(PhysicsFormula):
    """PV = nRT (理想氣體定律，簡化版 R=1)"""

    def __init__(self):
        super().__init__("P=nT/V")
        self.R = 1.0  # 簡化氣體常數

    def compute(self, n, T, V):
        return n * self.R * T / (V + 1e-6)  # 計算壓力P

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        n = np.random.uniform(min_val, max_val, n_samples)  # 莫耳數
        T = np.random.uniform(min_val * 10, max_val * 50, n_samples)  # 溫度
        V = np.random.uniform(min_val, max_val, n_samples)  # 體積
        P = self.compute(n, T, V)

        inputs = np.stack([n, T, V], axis=1)
        targets = P.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class Pendulum(PhysicsFormula):
    """T = 2π * sqrt(L/g) (單擺週期，簡化版)"""

    def __init__(self):
        super().__init__("T=2π*sqrt(L/g)")
        self.g = 9.8  # 重力加速度

    def compute(self, L):
        return 2 * np.pi * np.sqrt(L / self.g)

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        L = np.random.uniform(min_val, max_val, n_samples)
        T = self.compute(L)

        inputs = L.reshape(-1, 1)
        targets = T.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)


class CoulombsLaw(PhysicsFormula):
    """F = k * q1 * q2 / r^2 (庫倫定律)"""

    def __init__(self):
        super().__init__("F=q1*q2/r^2")
        self.k = 1.0  # 簡化庫倫常數

    def compute(self, q1, q2, r):
        return self.k * q1 * q2 / (r**2 + 1e-6)

    def generate_data(self, n_samples, value_range=(0.1, 10.0)):
        min_val, max_val = value_range
        q1 = np.random.uniform(-max_val, max_val, n_samples)  # 電荷可正可負
        q2 = np.random.uniform(-max_val, max_val, n_samples)
        r = np.random.uniform(max(min_val, 0.5), max_val, n_samples)
        f = self.compute(q1, q2, r)

        inputs = np.stack([q1, q2, r], axis=1)
        targets = f.reshape(-1, 1)

        return torch.FloatTensor(inputs), torch.FloatTensor(targets)
