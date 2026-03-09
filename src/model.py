import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

"""
基于冠脉本身结构的模型
"""
class GCN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(node_dim, hidden_dim)
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
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        

    def forward(self, x, edge_index, edge_attr):
        # 先做节点消息传递
        x = self.lin1(x)
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

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, images):
        """
        images: [B, L, C, D, H, W]
        B = batch size
        L = control point 数量
        C = CT channel
        24 * 24 * 24 = cube
        """

        feat = self.cnn3d(images)

        # [B,64,16,1,1]
        feat = feat.squeeze(-1).squeeze(-1)

        # [B,64,16] -> [B,16,64]
        feat = feat.permute(0,2,1)

        lstm_out,_ = self.lstm(feat)

        # 取最后一步
        y = lstm_out[:,-1]

        y = self.fc(y)

        return y

    
def load_model(device, node_dim, edge_dim, hidden_dim, out_dim=5):
    # 输出类别 1-RCA、2-LAD、3-LCX、4-LM、5-Other（按边分类）
    model = GCN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    model.to(device)
    model.eval()
    return model