import torch
import os
from datetime import datetime
import torch.optim as optim
import numpy as np
from .dataset import ArteryDataset
from .dataset import split_dataset
from sklearn.metrics import accuracy_score
from .model import load_model
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------
#  配置参数
# -------------------------
log_dir = "./logs"
save_dir = "./checkpoints"  # 保存模型的路径
hidden_dim = 256
out_dim = 5
# -------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_dir, "../data/S-Test")
test_dataset = ArteryDataset(data_path=test_data_path, num_samples=40)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "best_validation_2026-04-05_00-46-22.pth"
model_path = os.path.join(base_dir, f"../checkpoints/{model_name}")

print("Loading best model for final test evaluation...")
batch = next(iter(test_dataset))
model = load_model(device=device, 
           node_dim=batch.x.shape[1], 
           edge_dim=batch.edge_attr.shape[1],
           hidden_dim=hidden_dim,
           out_dim=out_dim,
           ckpt_path=model_path,
           eval_mode=True,
           )
print(f"Load model: {model_name}")

# -------------------------
#  测试
# -------------------------
def test(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.edge_patch, data.x, data.edge_index, data.edge_attr, data.edge_patch_index)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.edge_y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    unique_labels = np.unique(all_labels)

    # 用混淆矩阵计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)  # 对角线/该类总样本数

    # average=None 直接返回每个类别的指标数组
    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=unique_labels)
    per_class_f1     = f1_score(all_labels, all_preds,     average=None, labels=unique_labels)

    overall = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "recall":   recall_score(all_labels, all_preds, average="macro"),
        "f1":       f1_score(all_labels, all_preds,     average="macro"),
    }

    per_class = {
        label: {
            "accuracy": per_class_accuracy[i],
            "recall": per_class_recall[i],
            "f1":     per_class_f1[i],
        }
        for i, label in enumerate(unique_labels)
    }

    return overall, per_class
# -------------------------
# Test Evaluation
# -------------------------
overall, per_class = test(model, test_dataset, device)
print(f" --- Validation ---")
print(f"整体 Accuracy: {overall['accuracy']:.4f} | Recall: {overall['recall']:.4f} | F1: {overall['f1']:.4f}")
for label, metrics in per_class.items():
    print(f"  标签 {label}: Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
