import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        # 机械信号通常包含高频噪声，较大的卷积核有助于抗噪
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class MechanicEncoder(nn.Module):
    def __init__(self, input_channels=1, base_filters=64, output_feature_dim=512):
        """
        Args:
            input_channels (int): 输入信号通道数，单轴振动为1，三轴为3。
            base_filters (int): 卷积层宽度的基数，越大模型越复杂。
            output_feature_dim (int): 最终编码出的潜在特征向量维度。
        """
        super(MechanicEncoder, self).__init__()
        self.in_channels = base_filters
        
        # 初始特征提取层
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 残差堆叠层：提取深层工况不变特征
        self.layer1 = self._make_layer(base_filters, base_filters, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_filters * 4, output_feature_dim, blocks=2, stride=2)

        # 关键点：自适应池化。无论输入多长，都压缩成 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: (Batch, input_channels, Length) -> Length 可以是 1024, 2048, 4096 等任意值
        Output: (Batch, output_feature_dim)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) # 展平为向量
        return x