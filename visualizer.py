import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from PIL import Image


class Visualizer:
    def __init__(self, save_dir="plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.loss_history = []
        self.epoch_history = []

    def plot_results(self, epoch, loss, predictions, targets, save=True):
        """繪製loss和預測結果對比圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss曲線
        self.loss_history.append(loss)
        self.epoch_history.append(epoch)

        ax1.plot(self.epoch_history, self.loss_history, "b-")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True)

        # 預測vs真實值
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        ax2.scatter(targets_np, predictions_np, alpha=0.5, s=10)

        # 繪製理想線 y=x
        min_val = min(targets_np.min(), predictions_np.min())
        max_val = max(targets_np.max(), predictions_np.max())
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")

        ax2.set_xlabel("True Values")
        ax2.set_ylabel("Predictions")
        ax2.set_title(f"Predictions vs True Values (Epoch {epoch})")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save:
            filename = self.save_dir / f"epoch_{epoch:04d}.png"
            plt.savefig(filename, dpi=100)
            print(f"圖表已儲存: {filename}")

        # plt.show()
        plt.close()

    def create_animation(self, output_name="training.gif", duration=200):
        files = sorted(self.save_dir.glob("epoch_*.png"))
        if not files:
            print("⚠️ 沒有找到圖片，請先用 plot_results 產生圖片。")
            return

        frames = [Image.open(f) for f in files]

        output_path = self.save_dir / output_name
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,  # 每張圖顯示時間 (毫秒)
            loop=0,  # 0 表示無限循環
        )
        print(f"✅ GIF 動畫已建立: {output_path}")
