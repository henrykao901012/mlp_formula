import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)

        # 從config讀取loss函數
        self.criterion = config.criterion_class(**config.criterion_params)

        # 從config讀取優化器
        self.optimizer = config.optimizer_class(
            model.parameters(), **config.optimizer_params
        )

    def train(self, train_data, test_data, formula, visualizer):
        train_inputs, train_targets = train_data
        test_inputs, test_targets = test_data

        # 移到設備
        train_inputs = train_inputs.to(self.device)
        train_targets = train_targets.to(self.device)
        test_inputs = test_inputs.to(self.device)
        test_targets = test_targets.to(self.device)

        # 建立DataLoader
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 計算可視化間隔
        vis_interval_epochs = max(1, int(self.config.epochs * self.config.vis_interval))

        for epoch in range(1, self.config.epochs + 1):
            # 訓練
            self.model.train()
            epoch_loss = 0

            for batch_inputs, batch_targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # 可視化檢查
            if (
                epoch % vis_interval_epochs == 0
                or epoch == 1
                or epoch == self.config.epochs
            ):
                self.model.eval()
                with torch.no_grad():
                    test_predictions = self.model(test_inputs)
                    test_loss = self.criterion(test_predictions, test_targets).item()

                print(
                    f"Epoch {epoch}/{self.config.epochs}, Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}"
                )

                visualizer.plot_results(
                    epoch=epoch,
                    loss=test_loss,
                    predictions=test_predictions,
                    targets=test_targets,
                    inputs=test_inputs,  # 新增
                    formula=formula,  # 新增
                    save=self.config.save_plots,
                )

        # gif
        visualizer.create_animation()

        return self.model
