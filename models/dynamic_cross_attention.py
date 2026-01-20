import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import PreNorm_qkv, Attention

class DynamicCrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['model']['dmml']['fuison_transformer']['input_dim']
        heads = args['model']['dmml']['fuison_transformer']['heads']
        dim_head = dim // heads
        dropout = 0.1
        
        # 模态置信度预测（输入：模态特征全局池化，输出：[0,1]置信度）
        self.confidence_predictor = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1),
            nn.Sigmoid()
        )
        # 跨模态注意力（保留PreNorm_qkv兼容原有结构）
        self.cross_attn = PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        # 特征融合权重学习
        self.fusion_weight = nn.Parameter(torch.ones(3)/3)  # 初始化均等权重

    def forward(self, h_hyper, h_d_list_last, h_l, h_a, h_v):
        """
        h_hyper: 超模态特征 [B, seq_len, dim]
        h_d_list_last: 语言主导特征 [B, seq_len, dim]
        h_l/h_a/h_v: 降噪后的语言/音频/视觉特征 [B, seq_len, dim]
        """
        B, seq_len, dim = h_l.shape  # 获取批次和序列长度
        
        # 1. 预测各模态置信度
        conf_l = self.confidence_predictor(h_l.mean(dim=1))  # [B,1]
        conf_a = self.confidence_predictor(h_a.mean(dim=1))  # [B,1]
        conf_v = self.confidence_predictor(h_v.mean(dim=1))  # [B,1]
        
        # 关键修复：扩展置信度维度到 [B, seq_len, 1]，匹配特征维度
        conf_l = conf_l.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, 1]
        conf_a = conf_a.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, 1]
        conf_v = conf_v.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, 1]
        
        # 2. 动态加权的模态特征融合（现在维度匹配，可以正常相乘）
        fusion_feat = (
            self.fusion_weight[0] * conf_l * h_l +
            self.fusion_weight[1] * conf_a * h_a +
            self.fusion_weight[2] * conf_v * h_v
        )
        
        # 3. 跨模态注意力（以h_d_list_last为query，融合特征为key/value）
        cross_feat = self.cross_attn(h_d_list_last, fusion_feat, fusion_feat)
        # 残差连接增强特征
        out = cross_feat + h_hyper
        
        return out
