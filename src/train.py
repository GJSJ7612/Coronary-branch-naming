import torch
from datetime import datetime
import torch.optim as optim
import numpy as np
from .dataset import ArteryDataset
from .dataset import split_dataset
from .model import load_model
from .model import FocalLoss
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------
#  配置参数
# -------------------------
log_dir = "./logs"
save_dir = "./checkpoints"  # 保存模型的路径
hidden_dim = 256
num_samples = 200
num_epochs = 50
val_size = 0.2
seed = 42
lr = 1e-4
min_lr = 1e-6  # CosineAnnealing 的最小学习率
# -------------------------

dataset = ArteryDataset(num_samples=num_samples)

# 分割数据集
train_set, val_set = split_dataset(dataset, val_size=val_size, seed=seed)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = next(iter(train_loader))
model = load_model(device=device, 
           node_dim=batch.x.shape[1], 
           edge_dim=batch.edge_attr.shape[1],
           hidden_dim=hidden_dim,
           out_dim=5,
           eval_mode=False)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.8, 2.3, 3.0, 0.9]).to(device))
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# 创建保存模型的文件夹
import os
os.makedirs(save_dir, exist_ok=True)
best_val_acc = 0.0  # 保存历史最好的验证精度

# -------------------------
# 训练 & 测试 函数
# -------------------------
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.edge_patch, data.x, data.edge_index, data.edge_attr, data.edge_patch_index)
        loss = criterion(out, data.edge_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)

    return avg_loss

def vaild(model, data_loader, device):
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

    # 整体准确率
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # 各标签准确率
    unique_labels = np.unique(all_labels)
    per_class_accuracy = {}
    for label in unique_labels:
        mask = all_labels == label
        per_class_accuracy[label] = accuracy_score(all_labels[mask], all_preds[mask])

    return overall_accuracy, per_class_accuracy

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
# Training Loop
# -------------------------
print("Starting training...")
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, device)

    # cosine annealing update lr
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

    if (epoch + 1) % 10 == 0:
        overall_acc, per_class_acc = vaild(model, val_loader, device)
        print(f" --- Validation ---")
        print(f"整体准确率: {overall_acc:.4f}")
        for label, acc in per_class_acc.items():
            print(f"  标签 {label}: {acc:.4f}")
        
        # 保存验证集最佳模型
        if overall_acc > best_val_acc:
            best_val_acc = overall_acc
            
            best_model_path = os.path.join(save_dir, f"best_validation_{time_str}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1}, acc={overall_acc:.4f}")

final_model_path = os.path.join(save_dir, f"final_{time_str}.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Training complete. Final model saved: {final_model_path}")