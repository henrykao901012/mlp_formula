import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # 使用非互動式後端


class Visualizer:
    def __init__(self, save_dir="plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.loss_history = []
        self.epoch_history = []

    def plot_results(
        self, epoch, loss, predictions, targets, inputs, formula, save=True
    ):
        """繪製loss和預測結果對比圖 - 統一2D版本"""
        plt.close("all")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        inputs_np = inputs.detach().cpu().numpy().flatten()
        predictions_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # 圖1: Loss曲線
        ax1 = axes[0]
        self.loss_history.append(loss)
        self.epoch_history.append(epoch)

        ax1.plot(self.epoch_history, self.loss_history, "b-", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)

        # 使用log scale如果loss變化很大
        if len(self.loss_history) > 1:
            max_loss = max(self.loss_history)
            min_loss = min(self.loss_history)
            if min_loss > 0 and max_loss / min_loss > 100:
                ax1.set_yscale("log")

        # 圖2: Input vs Output (主要圖形)
        ax2 = axes[1]

        # 繪製NN預測點
        ax2.scatter(
            inputs_np,
            predictions_np,
            alpha=0.6,
            s=30,
            label="NN Predictions",
            color="blue",
            zorder=5,
        )

        # 繪製真實公式曲線（虛線）
        x_range = np.linspace(inputs_np.min(), inputs_np.max(), 200)
        y_true = np.array([formula.compute(x) for x in x_range])
        ax2.plot(
            x_range,
            y_true,
            "r--",
            linewidth=2,
            label=f"True: {formula.get_description()}",
            alpha=0.8,
        )

        ax2.set_xlabel(f"{formula.variable_name}", fontsize=12)
        ax2.set_ylabel(f"{formula.output_name}", fontsize=12)
        ax2.set_title(f"Input vs Output - Epoch {epoch}")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        # 圖3: 預測vs真實值
        ax3 = axes[2]
        ax3.scatter(targets_np, predictions_np, alpha=0.5, s=20, color="green")

        # 繪製理想線 y=x
        min_val = min(targets_np.min(), predictions_np.min())
        max_val = max(targets_np.max(), predictions_np.max())
        ax3.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
            alpha=0.8,
        )

        # 計算並顯示R²分數
        ss_res = np.sum((predictions_np - targets_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 計算RMSE
        rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))

        ax3.set_xlabel("True Values")
        ax3.set_ylabel("Predictions")
        ax3.set_title(f"Predictions vs True\nR²={r2_score:.4f}, RMSE={rmse:.4f}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 設置整體標題
        fig.suptitle(
            f"Training Progress: {formula.get_description()}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save:
            filename = self.save_dir / f"epoch_{epoch:04d}.png"
            try:
                plt.savefig(filename, dpi=100, bbox_inches="tight")
                print(f"圖表已儲存: {filename}")
            except Exception as e:
                print(f"儲存圖片失敗: {e}")

        plt.close("all")

    def create_animation(self, output_name="training.gif", duration=200):
        files = sorted(self.save_dir.glob("epoch_*.png"))
        if not files:
            print("⚠️ 沒有找到圖片，請先用 plot_results 產生圖片。")
            return

        try:
            frames = [Image.open(f) for f in files]
            output_path = self.save_dir / output_name
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
            print(f"✅ GIF 動畫已建立: {output_path}")
        except Exception as e:
            print(f"❌ GIF 動畫建立失敗: {e}")
