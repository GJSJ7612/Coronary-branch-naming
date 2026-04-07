"""
pipeline/predict.py
───────────────────
把模型推理结果（边级标签）转换为体素级 labeled_mask，
供 main.py 的 /api/predict 接口调用。

对外只暴露一个函数：
    labeled_mask = run_pipeline(ccta_path, mask_path, model_cfg)
"""
import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

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

# ── 把你的项目根目录加入 sys.path ──────────────────────────────────
# 假设目录结构：
#   ccta_system/
#     backend/
#       main.py
#       pipeline/
#         predict.py        ← 本文件
#     your_project/         ← 你原有代码所在的目录
#       dataset.py
#       model.py
#       data_process.py
#       ...

# ── 导入你项目的模块 ───────────────────────────────────────────────
# 把下面的模块名换成你实际的文件/包名
from src.data_process import data_process          # 图构建函数
from src.dataset import OffsetData                 # 你的 Data 子类
from src.model import load_model                   # 你的模型加载函数


# ─────────────────────────────────────────────────────────────────
# 1. 把图+预测标签 → 体素级 labeled_mask
# ─────────────────────────────────────────────────────────────────
def graph_preds_to_labeled_mask(G, preds, mask_shape, original_mask):
    from scipy.spatial import cKDTree

    raw_mask_binary = (original_mask > 0).copy()
    
    # 调试：原始mask信息
    print(f"original_mask 唯一值: {np.unique(original_mask)}")
    print(f"original_mask 非零体素数: {raw_mask_binary.sum()}")
    print(f"original_mask shape: {original_mask.shape}")
    
    center_points = []
    center_labels = []
    for edge_id, (u, v) in enumerate(G.edges()):
        edge_label = int(preds[edge_id]) + 1
        for (xi, yi, zi) in G.edges[u, v].get("pixels", []):
            center_points.append([xi, yi, zi])
            center_labels.append(edge_label)

    print(f"中心线点总数: {len(center_points)}")
    print(f"中心线标签分布: {np.bincount(center_labels)}")
    
    center_points = np.array(center_points)
    center_labels = np.array(center_labels)
    tree = cKDTree(center_points)

    block_size = 64
    final_labeled = np.zeros(mask_shape, dtype=np.uint8)

    for z_start in range(0, mask_shape[0], block_size):
        for y_start in range(0, mask_shape[1], block_size):
            for x_start in range(0, mask_shape[2], block_size):
                z_end = min(z_start + block_size, mask_shape[0])
                y_end = min(y_start + block_size, mask_shape[1])
                x_end = min(x_start + block_size, mask_shape[2])

                z, y, x = np.meshgrid(
                    np.arange(z_start, z_end),
                    np.arange(y_start, y_end),
                    np.arange(x_start, x_end),
                    indexing='ij'
                )
                coords = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1)
                valid_mask = raw_mask_binary[coords[:,0], coords[:,1], coords[:,2]]
                valid_coords = coords[valid_mask]
                if len(valid_coords) == 0:
                    continue
                _, idx = tree.query(valid_coords, k=1)
                final_labeled[
                    valid_coords[:,0],
                    valid_coords[:,1],
                    valid_coords[:,2]
                ] = center_labels[idx]

    # 调试：填充结果
    print(f"final_labeled 唯一值: {np.unique(final_labeled)}")
    print(f"final_labeled 各label体素数: {np.bincount(final_labeled.ravel())}")
    
    return final_labeled

# ─────────────────────────────────────────────────────────────────
# 2. 构建单样本的 OffsetData（复用 dataset.get() 的逻辑）
# ─────────────────────────────────────────────────────────────────

def build_graph_data(ccta_path: str, mask_path: str):
    """
    用你的 data_process 函数从文件路径构建图，
    然后组装成 OffsetData 对象（与 ArteryDataset.get() 完全一致）。

    返回：(data: OffsetData, G: networkx.Graph)
    """
    G = data_process(mask_path, ccta_path)

    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 节点特征
    node_pos = [G.nodes[i]["pos"] for i in G.nodes()]
    x = torch.tensor(node_pos, dtype=torch.float)

    # 边特征和标签
    edge_attrs, edge_labels = [], []
    for (u, v) in edges:
        d = G.edges[u, v]
        feat = np.hstack([d["node_3d"], d["node_SCT"], d["z_axis"], d["y_axis"]])
        edge_attrs.append(feat)
        edge_labels.append(int(d.get("label", -1)))

    edge_attr = torch.from_numpy(np.array(edge_attrs, dtype=np.float32)).squeeze(1)
    edge_y    = torch.tensor(edge_labels, dtype=torch.long)

    data = OffsetData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_y=edge_y,
    )
    data.graph = G

    # 图像 patch 特征
    patches, edge_idx = [], []
    for edge_id, (u, v) in enumerate(edges):
        images = G.edges[u, v].get("image", [])
        for patch in images:
            patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # [1,D,H,W]
            patches.append(patch_t)
            edge_idx.append(edge_id)

    data.edge_patch       = torch.stack(patches)          # [N,1,D,H,W]
    data.edge_patch_index = torch.tensor(edge_idx)        # [N]

    return data, G


# ─────────────────────────────────────────────────────────────────
# 3. 对外接口：run_pipeline
# ─────────────────────────────────────────────────────────────────

# 模型配置（按你的实际参数填写）
MODEL_CFG = {
    "ckpt_path":  "checkpoints/best_validation_2026-03-31_14-49-05.pth",  # ← 改成你的权重路径
    "hidden_dim": 256,
    "out_dim":    5,      # LM / LAD / LCX / RCA / Other
}

# 模型单例缓存（避免每次请求都重新加载权重）
_model_cache: dict = {}

def _get_model(device: torch.device, node_dim: int, edge_dim: int):
    key = str(device)
    if key not in _model_cache:
        model = load_model(
            device=device,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=MODEL_CFG["hidden_dim"],
            out_dim=MODEL_CFG["out_dim"],
            ckpt_path=MODEL_CFG["ckpt_path"],
            eval_mode=True,
        )
        _model_cache[key] = model
    return _model_cache[key]


def run_pipeline(ccta_path: str, mask_path: str) -> np.ndarray:
    """
    完整推理流程：
      文件路径 → 图构建 → 模型推理 → 体素级 labeled_mask

    返回：np.ndarray，形状与 mask 文件相同，
          值为 0(背景) / 1(LM) / 2(LAD) / 3(LCX) / 4(RCA) / 5(Other)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理使用设备: {device}")

    # 1. 构建图数据
    data, G = build_graph_data(ccta_path, mask_path)
    print("图构建完成，节点数:", data.x.shape[0], "边数:", data.edge_index.shape[1])

    # 2. 加载模型（首次调用时才真正加载）
    model = _get_model(
        device=device,
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
    )
    print("模型加载完成，开始推理...")
    
    # 3. 推理
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out  = model(
            data.edge_patch,
            data.x,
            data.edge_index,
            data.edge_attr,
            data.edge_patch_index,
        )
        preds = out.argmax(dim=1).cpu().numpy()   # shape: [num_edges]
    print("模型推理完成，正在生成 labeled_mask...")

    # 4. 获取原始 mask 形状（用于写回体素）
    from main import load_volume
    raw_mask = load_volume(mask_path)
    mask_shape = raw_mask.shape

    # 5. 边级标签 → 体素级 labeled_mask
    labeled = graph_preds_to_labeled_mask(G, preds, mask_shape, raw_mask)
    print("labeled_mask 生成完成。")

    return labeled
