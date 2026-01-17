import torch
import torch.nn as nn
import torch.nn.functional as F

class MechanicDecoder(nn.Module):
    def __init__(self, feature_dim=512, output_channels=1, base_filters=64):
        """
        Args:
            feature_dim (int): 必须与 Encoder 的 output_feature_dim 一致。
            output_channels (int): 必须与 Encoder 的 input_channels 一致。
            base_filters (int): 控制解码器的宽度。
        """
        super(MechanicDecoder, self).__init__()
        
        # 这种设计是对称的，旨在将特征向量逐步放大
        self.init_conv = nn.Linear(feature_dim, base_filters * 8 * 4) # 先映射到一个小的长度 (4)
        self.reshape_dim = base_filters * 8
        
        # 上采样块
        self.up1 = self._make_up_block(base_filters * 8, base_filters * 4)
        self.up2 = self._make_up_block(base_filters * 4, base_filters * 2)
        self.up3 = self._make_up_block(base_filters * 2, base_filters)
        
        # 最终输出层
        self.final_conv = nn.Conv1d(base_filters, output_channels, kernel_size=3, padding=1)

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
            # 注意：上采样逻辑在 forward 中动态处理
        )

    def forward(self, z, target_length=1024):
        """
        Args:
            z (Tensor): 潜在特征向量 (Batch, feature_dim)
            target_length (int): 期望还原的原始信号长度 (例如 1024, 2048)。
                               在训练循环中，你可以直接传入 x.shape[-1]
        """
        # 1. 将向量映射回张量
        x = self.init_conv(z)
        # Reshape 为 (Batch, Channel, Length=4)
        x = x.view(x.size(0), self.reshape_dim, 4) 
        
        # 2. 逐步上采样。使用插值法(Interpolate)比反卷积更灵活，适应任意长度
        # 每次放大2倍，最后一次强制对齐到 target_length
        
        x = F.interpolate(x, scale_factor=4, mode='linear', align_corners=False) # L=16
        x = self.up1(x)
        
        x = F.interpolate(x, scale_factor=4, mode='linear', align_corners=False) # L=64
        x = self.up2(x)
        
        x = F.interpolate(x, scale_factor=4, mode='linear', align_corners=False) # L=256
        x = self.up3(x)
        
        # 3. 最终强制对齐到目标长度
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        x = self.final_conv(x)
        return x # (Batch, output_channels, target_length)