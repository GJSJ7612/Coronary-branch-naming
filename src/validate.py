import torch
import torch.optim as optim
import numpy as np
from dataset import ArteryDataset
from sklearn.metrics import accuracy_score
from model import load_model
from torch_geometric.loader import DataLoader
from data_process import plot_graph_3d_with_label
import random

# -------------------------
#  配置参数
# -------------------------
log_dir = "./logs"
model_path = "./checkpoints/best_validation_2026-03-13_23-42-34.pth"  # 模型权重路径
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
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = next(iter(data_loader))
model = load_model(device=device, 
           node_dim=batch.x.shape[1], 
           edge_dim=batch.edge_attr.shape[1],
           hidden_dim=hidden_dim,
           out_dim=5,
           ckpt_path=model_path,
           eval_mode=True)

def validate(model, dataset, device):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    sample = dataset[idx]

    with torch.no_grad():
        sample = sample.to(device)
        out = model(sample.edge_patch, sample.x, sample.edge_index, sample.edge_attr, sample.edge_patch_index)
        preds = out.argmax(dim=1)

    print("preds:", preds.cpu().numpy())
    accuracy = accuracy_score(sample.edge_y.cpu().numpy(), preds.cpu().numpy())
    print(f"Validation Accuracy: {accuracy:.4f}")
    plot_graph_3d_with_label(sample.graph, preds)

    return accuracy

print("Starting validation...")
val_acc = validate(model, dataset, device)
