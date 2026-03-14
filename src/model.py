import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

"""
基于冠脉本身结构的模型
"""
class ConditionalGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, cond_dim):
        super().__init__()
        self.lin1 = nn.Linear(node_dim, hidden_dim)

        # condition modulation
        self.gamma = nn.Linear(cond_dim, edge_dim)
        self.beta  = nn.Linear(cond_dim, edge_dim)

        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)
        
        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)

        self.conv3 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), edge_dim=edge_dim)

        self.edge_cls = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        
    def condition(self, x, y):
        """
        FiLM conditioning
        """
        gamma = self.gamma(y)
        beta  = self.beta(y)

        return gamma * x + beta

    def forward(self, x, edge_index, edge_attr, cond):
        x = self.lin1(x)
        edge_attr = self.condition(edge_attr, cond)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # 按 edge 取两端节点嵌入，和 edge_attr 拼接做边分类
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([x[src], x[dst], edge_attr], dim=-1)

        return self.edge_cls(edge_feat)
    
"""
基于影像的模型
"""
class ConditionExtractor(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()

        # 3D CNN
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size = 64,
            hidden_size = 128,
            num_layers = 4,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, edge_num, images, index):
        """
        images: [E*L, C, D, H, W]
        E = edge num
        L = control point num
        C = CT channel
        D = cube depth
        H = cube height
        W = cube width
        """

        feat = self.cnn3d(images)  # [E*L, 64, D', H', W']
        feat = torch.nn.functional.adaptive_avg_pool3d(feat, (1, 1, 1)) # [E*L, 64, 1, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [E*L, 64]

        edge_groups = []
        edge_lengths = []
        for i in range(edge_num): # B * E
            mask = (index == i)
            
            edge_feat = feat[mask] # 提取第 i 条边的所有 patch 特征
            edge_groups.append(edge_feat)
            edge_lengths.append(edge_feat.size(0)) # 记录实际长度

        padded_seqs = pad_sequence(edge_groups, batch_first=True)          # [E, L_max, 64]
        lengths_tensor = torch.tensor(edge_lengths, dtype=torch.long, device=images.device)
        packed = pack_padded_sequence(padded_seqs, lengths_tensor.cpu(),
                                      batch_first=True, enforce_sorted=False)

        _, (hn, _) = self.lstm(packed)
        # hn shape: [num_layers*2, E, hidden_dim]
        # 取最后一层的前向(hn[-2])和后向(hn[-1])拼接
        cond = torch.cat([hn[-2], hn[-1]], dim=-1)  # [E, hidden_dim*2] = [E, 256]
        return cond

class FullModel(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim, cond_dim):
        super().__init__()

        self.condition_extractor = ConditionExtractor(
            in_channels=1,
            hidden_dim=hidden_dim
        )

        self.gnn = ConditionalGNN(
            node_dim,
            edge_dim,
            hidden_dim,
            out_dim,
            cond_dim
        )

    def forward(self, images, x, edge_index, edge_attr, edge_patch_index):

        # 提取条件
        cond = self.condition_extractor(edge_index.size(1), images, edge_patch_index)

        # GNN
        out = self.gnn(x, edge_index, edge_attr, cond)

        return out
    
# 输出类别 0-RCA、1-LAD、2-LCX、3-LM、4-Other（按边分类）    
def load_model(
    device,
    node_dim,
    edge_dim,
    hidden_dim,
    out_dim=5,
    ckpt_path=None,
    eval_mode=True
):
    
    model = FullModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,

        # 注：在GCN中的 hidden_dim 是 256，
        #     condition extractor 输出是 128，
        #     所以这里直接用 hidden_dim 作为 cond_dim，保持一致即可
        cond_dim=hidden_dim 
    )
    model = model.to(device)

    # 如果需要加载权重
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state)

    if eval_mode:
        model.eval()
    else:
        model.train()

    return model