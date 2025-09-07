import torch


class Config:
    # 模型參數
    input_dim = 3  # 會根據公式自動調整
    hidden_dims = [64, 32, 16]  # 增加網路深度處理複雜公式
    output_dim = 1

    # 訓練參數
    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # 資料參數
    train_samples = 1000
    test_samples = 200
    data_range = (0.1, 10.0)  # m和a的範圍

    # 可視化參數
    vis_interval = 0.05  # 可視化間隔比例
    save_plots = True
    plot_dir = "plots"

    # 其他
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
