import torch
import torch.nn as nn
import torch.optim as optim


class Config:
    # 擬合公式
    # 可選公式: "NewtonSecondLaw", "KineticEnergy", "GravitationalForce",
    #          "IdealGasLaw", "Pendulum", "CoulombsLaw", "HookesLaw"
    formula = "IdealGasLaw"

    # 模型參數（現在統一為單輸入單輸出）
    input_dim = 1
    hidden_dims = [32, 16, 8]  # 簡化網路結構
    output_dim = 1

    # 訓練參數
    epochs = 200
    batch_size = 32

    # 優化器設定
    optimizer_class = optim.Adam
    optimizer_params = {"lr": 0.001, "weight_decay": 0}

    # Loss函數設定
    criterion_class = nn.MSELoss
    criterion_params = {}

    # 資料參數
    train_samples = 1000
    test_samples = 200
    data_range = (0.1, 10.0)

    # 可視化參數
    vis_interval = 0.1  # 每10%的epoch畫一次圖
    save_plots = True
    plot_dir = "plots"

    # 其他
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
