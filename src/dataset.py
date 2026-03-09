import os
import torch
import numpy as np
from torch_geometric.data import Dataset
from data_process import data_process
from torch_geometric.data import Data

class ArteryDataset(Dataset):
    def __init__(self, data_path="/home/gjsj/project/undergraduate/data/Input", num_samples=100):
        super().__init__(None)
        self.data_path = data_path
        self.num_samples = num_samples

    def len(self):
        return self.num_samples

    def get(self, idx):
        # 读取数据与label
        image_path = os.path.join(self.data_path, f"{idx}.img.nii.gz")
        data_path = os.path.join(self.data_path, f"{idx}.label.nrrd")
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
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_y=edge_y,
        )

        assert edge_index.max() < x.size(0), "edge_index 越界"
        assert edge_index.min() >= 0, "出现负编号"
        return data

def split_dataset(dataset, test_size=0.15, val_size=0.15, seed=42):
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

    num_samples = len(dataset)
    indices = list(range(num_samples))

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size / (1 - test_size), random_state=seed, shuffle=True
    )

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    return train_set, val_set, test_set
