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
    HookesLaw,
)
from trainer import Trainer
from visualizer import Visualizer


# 映射表：字串 → 公式建構函數
formula_map = {
    "NewtonSecondLaw": lambda: NewtonSecondLaw(m=2.0),
    "KineticEnergy": lambda: KineticEnergy(m=1.0),
    "GravitationalForce": lambda: GravitationalForce(m1=5.0, m2=10.0),
    "IdealGasLaw": lambda: IdealGasLaw(n=1.0, T=300.0),
    "Pendulum": lambda: Pendulum(),
    "CoulombsLaw": lambda: CoulombsLaw(q1=2.0, q2=3.0),
    "HookesLaw": lambda: HookesLaw(k=10.0),
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    # 載入配置
    config = Config()
    set_seed(config.seed)

    # 從 Config 直接選擇
    try:
        formula = formula_map[Config.formula]()
    except KeyError:
        raise ValueError(f"Unknown formula: {Config.formula}")

    print(f"學習公式: {formula.get_description()}")
    print(f"輸入變數: {formula.variable_name}")
    print(f"輸出變數: {formula.output_name}")

    # 所有公式現在都是單輸入單輸出
    config.input_dim = 1
    config.output_dim = 1

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
    trained_model = trainer.train(train_data, test_data, formula, visualizer)

    # 測試預測
    print("\n測試預測:")
    test_values = [1.0, 2.5, 5.0, 7.5, 10.0]

    for val in test_values:
        input_tensor = torch.FloatTensor([[val]]).to(config.device)
        prediction = model(input_tensor).item()
        true_value = formula.compute(val)
        error = abs(prediction - true_value)
        print(
            f"{formula.variable_name}={val:.2f}: "
            f"預測{formula.output_name}={prediction:.3f}, "
            f"真實{formula.output_name}={true_value:.3f}, "
            f"誤差={error:.3f}"
        )


if __name__ == "__main__":
    main()
