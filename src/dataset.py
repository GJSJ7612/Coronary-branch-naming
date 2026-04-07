import os
import torch
import numpy as np
from torch_geometric.data import Dataset
from .data_process import data_process
from torch_geometric.data import Data
from collections import Counter

class OffsetData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_patch_index':
            # 每次 batch 合并时，该字段加上当前图的边数
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # edge_patch_index 沿 dim=0 拼接（默认行为，显式声明更安全）
        if key == 'edge_patch':
            return 0   # 沿 batch 维度拼接，shape [N,1,D,H,W] → dim=0 是N
        return super().__cat_dim__(key, value, *args, **kwargs)

class ArteryDataset(Dataset):
    def __init__(self, data_path=None, num_samples=100):
        super().__init__(None)
        if data_path is None:
            # 相对于当前脚本文件的路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "../data/S-Input")
        self.data_path = data_path
        self.num_samples = num_samples

    def len(self):
        return self.num_samples

    def get(self, idx):
        # 读取数据与label
        image_path = os.path.join(self.data_path, f"{idx+1}.img.nii.gz")
        data_path = os.path.join(self.data_path, f"{idx+1}.label.nrrd")
        G = data_process(data_path, image_path)

        # ======== 1. 构建 edge_index ========
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # ======== 2. 构建 node 特征 x ========
        node_pos = []
        for i in G.nodes():
            node_pos.append(G.nodes[i]["pos"])  # 节点位置
        x = torch.tensor(node_pos, dtype=torch.float)   # [num_nodes, 3]

        # ======== 3. 构建 edge_attr ========
        edge_attrs = []
        edge_labels = []
        for (u, v) in edges:
            data = G.edges[u, v]

            feat = np.hstack([data["node_3d"], data["node_SCT"], data["z_axis"], data["y_axis"]])
            edge_attrs.append(feat)
            edge_labels.append(int(data.get("label", -1)))

        edge_attr = torch.from_numpy(np.array(edge_attrs, dtype=np.float32)).squeeze(1)
        edge_y = torch.tensor(edge_labels, dtype=torch.long)

        # ======== 4. 构建 Data 对象 ========
        data = OffsetData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_y=edge_y,
        )
        data.graph = G

        # ======== 5. 构建 edge 的图像特征 ========
        patches = []
        edge_idx = []

        for edge_id, (u, v) in enumerate(edges):

            edge = G.edges[u, v]
            images = edge.get("image") # [L, D, H, W]

            for patch in images:

                patch = torch.tensor(patch, dtype=torch.float32)
                patch = patch.unsqueeze(0)   # [1,D,H,W]

                patches.append(patch)
                edge_idx.append(edge_id)
        
        patch_tensor = torch.stack(patches)  # [N,1,D,H,W]
        edge_idx = torch.tensor(edge_idx)    # [N]

        data.edge_patch = patch_tensor
        data.edge_patch_index = edge_idx

        assert edge_index.max() < x.size(0), "edge_index 越界"
        assert edge_index.min() >= 0, "出现负编号"
        return data

def split_dataset(dataset, val_size=0.2, seed=42):
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    num_samples = len(dataset)
    indices = list(range(num_samples))

    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, shuffle=True
    )

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    return train_set, val_set

# def compute_edge_class_weights(dataset, num_classes=None):
#     """
#     遍历 dataset，统计所有边的标签分布，返回类别权重 tensor
#     """
#     all_labels = []
    
#     print("正在统计标签分布...")
#     for i in range(len(dataset)):
#         data = dataset[i]
#         labels = data.edge_y  # [num_edges]
#         # 过滤掉无效标签（如 -1）
#         valid_mask = labels >= 0
#         all_labels.append(labels[valid_mask])
    
#     all_labels = torch.cat(all_labels, dim=0).numpy()  # [total_edges]
    
#     # 统计每类数量
#     counter = Counter(all_labels)
#     print(f"标签分布: {dict(sorted(counter.items()))}")
    
#     if num_classes is None:
#         num_classes = max(counter.keys()) + 1
    
#     # 计算权重：总样本数 / (类别数 × 该类样本数)  — sklearn 的 balanced 策略
#     total = len(all_labels)
#     weights = []
#     for c in range(num_classes):
#         count = counter.get(c, 1)  # 避免除零
#         w = total / (num_classes * count)
#         weights.append(w)
    
#     weights = torch.tensor(weights, dtype=torch.float32)
#     print(f"类别权重: {weights}")
#     return weights