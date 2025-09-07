import torch
import numpy as np
from config import Config
from model import MLP
from formula import (
    NewtonSecondLaw,
    KineticEnergy,
    GravitationalForce,
    IdealGasLaw,
    Pendulum,
    CoulombsLaw,
)
from trainer import Trainer
from visualizer import Visualizer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    # 載入配置
    config = Config()
    set_seed(config.seed)

    # 選擇要學習的公式
    # formula = NewtonSecondLaw()  # F=ma (2輸入)
    # formula = KineticEnergy()  # KE=0.5*m*v^2 (2輸入)
    formula = GravitationalForce()  # F=m1*m2/r^2 (3輸入)
    # formula = IdealGasLaw()  # P=nT/V (3輸入)
    # formula = Pendulum()  # T=2π*sqrt(L/g) (1輸入)
    # formula = CoulombsLaw()  # F=q1*q2/r^2 (3輸入)

    print(f"學習公式: {formula.name}")

    # 根據公式自動調整輸入維度
    if isinstance(formula, (NewtonSecondLaw, KineticEnergy)):
        config.input_dim = 2
    elif isinstance(formula, (GravitationalForce, IdealGasLaw, CoulombsLaw)):
        config.input_dim = 3
    elif isinstance(formula, Pendulum):
        config.input_dim = 1

    # 生成資料
    train_data = formula.generate_data(config.train_samples, config.data_range)
    test_data = formula.generate_data(config.test_samples, config.data_range)

    # 建立模型
    model = MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    )

    # 初始化視覺化工具
    visualizer = Visualizer(save_dir=config.plot_dir)

    # 訓練
    trainer = Trainer(model, config)
    trained_model = trainer.train(train_data, test_data, visualizer)

    # 測試預測
    print("\n測試預測:")
    if isinstance(formula, NewtonSecondLaw):
        test_cases = [(1.0, 2.0), (5.0, 3.0), (2.5, 9.8)]
        for m, a in test_cases:
            inputs = torch.FloatTensor([[m, a]]).to(config.device)
            prediction = model(inputs).item()
            true_value = formula.compute(m, a)
            error = abs(prediction - true_value)
            print(
                f"m={m}, a={a}: 預測F={prediction:.3f}, 真實F={true_value:.3f}, 誤差={error:.3f}"
            )

    elif isinstance(formula, GravitationalForce):
        test_cases = [(1.0, 2.0, 1.0), (5.0, 3.0, 2.0), (10.0, 10.0, 5.0)]
        for m1, m2, r in test_cases:
            inputs = torch.FloatTensor([[m1, m2, r]]).to(config.device)
            prediction = model(inputs).item()
            true_value = formula.compute(m1, m2, r)
            error = abs(prediction - true_value)
            print(
                f"m1={m1}, m2={m2}, r={r}: 預測F={prediction:.3f}, 真實F={true_value:.3f}, 誤差={error:.3f}"
            )


if __name__ == "__main__":
    main()
