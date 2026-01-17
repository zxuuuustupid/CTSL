import torch
import torch.nn as nn

class MechanicClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=10, dropout_rate=0.5):
        """
        Args:
            feature_dim (int): 必须与 Encoder 的输出维度一致。
            num_classes (int): 故障类别总数（例如 10类：正常 + 3部位*3尺寸）。
                             这个数字在实例化时从数据集读取。
            dropout_rate (float): 防止过拟合。
        """
        super(MechanicClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
            # 注意：这里不加 Softmax，因为 PyTorch 的 CrossEntropyLoss 自带 Softmax
        )
        
        # 权重初始化（对深度网络很重要）
        self._initialize_weights()

    def forward(self, x):
        """
        Input: (Batch, feature_dim) - 来自 Encoder 的输出
        Output: (Batch, num_classes) - Logits
        """
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)