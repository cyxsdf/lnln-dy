import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseAdaptiveSuppression(nn.Module):
    def __init__(self, args, modal_dim):
        super().__init__()
        self.modal_dim = modal_dim
        # 噪声比例预测层（输出∈[0,1]，表示噪声占比）
        self.noise_predictor = nn.Sequential(
            nn.Linear(modal_dim, modal_dim // 2),
            nn.ReLU(),
            nn.Linear(modal_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, modal_feat):
        # modal_feat: [B, seq_len, modal_dim]
        B, seq_len, _ = modal_feat.shape
        
        # 1. 全局池化得到样本级特征，用于预测噪声比例
        global_feat = modal_feat.mean(dim=1)  # [B, modal_dim]
        noise_ratio = self.noise_predictor(global_feat)  # [B, 1]
        
        # 2. 自适应抑制噪声：噪声比例越高，特征缩放越明显
        suppress_weight = 1 - noise_ratio  # [B, 1]
        suppress_weight = suppress_weight.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, 1]
        
        # 3. 降噪后的特征
        denoised_feat = modal_feat * suppress_weight
        
        return denoised_feat

# 多模态噪声自适应抑制封装
class MultiModalNoiseSuppression(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.l_dim = args['model']['feature_extractor']['hidden_dims'][0]
        self.a_dim = args['model']['feature_extractor']['hidden_dims'][2]
        self.v_dim = args['model']['feature_extractor']['hidden_dims'][1]
        
        self.l_suppress = NoiseAdaptiveSuppression(args, self.l_dim)
        self.a_suppress = NoiseAdaptiveSuppression(args, self.a_dim)  # 音频噪声抑制（重点）
        self.v_suppress = NoiseAdaptiveSuppression(args, self.v_dim)

    def forward(self, h_l, h_a, h_v):
        h_l_denoised = self.l_suppress(h_l)
        h_a_denoised = self.a_suppress(h_a)  # 重点抑制音频噪声
        h_v_denoised = self.v_suppress(h_v)
        
        return h_l_denoised, h_a_denoised, h_v_denoised
