import torch
import torch.optim as optim
import numpy as np
from dataset import ArteryDataset
from dataset import split_dataset
from sklearn.metrics import accuracy_score
from model import load_model
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# -------------------------
#  配置参数
# -------------------------
log_dir = "./logs"
save_dir = "./checkpoints"  # 保存模型的路径
hidden_dim = 256
num_samples = 100
num_epochs = 10000
test_size = 0.15
val_size = 0.15
seed = 42
lr = 1e-4
min_lr = 1e-6  # CosineAnnealing 的最小学习率
# -------------------------

dataset = ArteryDataset()
# writer = SummaryWriter(log_dir)

# 分割数据集
train_set, val_set, test_set = split_dataset(dataset, test_size, val_size, seed)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = next(iter(train_loader))
model = load_model(device=device, 
           node_dim=batch.x.shape[1], 
           edge_dim=batch.edge_attr.shape[1],
           hidden_dim=hidden_dim,
           out_dim=5)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

# 创建保存模型的文件夹
import os
os.makedirs(save_dir, exist_ok=True)
best_val_acc = 0.0  # 保存历史最好的验证精度

# -------------------------
# 训练 & 测试 函数
# -------------------------
def train(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    epoch_loss = 0

    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.edge_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)

    return avg_loss

def test(model, data_loader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.edge_y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy

# -------------------------
# Training Loop
# -------------------------
print("Starting training...")
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, device, epoch)

    # cosine annealing update lr
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

    if (epoch + 1) % 10 == 0:
        val_acc = test(model, val_loader, device, epoch)
        print(f" --- Validation Accuracy: {val_acc:.4f}")

        # 保存验证集最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, "best_validation.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1}, acc={val_acc:.4f}")

final_model_path = os.path.join(save_dir, "final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Training complete. Final model saved: {final_model_path}")