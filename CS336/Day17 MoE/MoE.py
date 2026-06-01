import torch
import torch.nn as nn
import torch.nn.functional as F

from model_moe import Linear
from model_moe import SwiGLU
from model_moe import softmax

# 假设有n个专家，那么g = softmax(x @ w_g) -> [batch_size,seq,n]
# 其中w_g的维度是[batch_size,d_model,n]
# 每一个元素代表了 Router 认为把该 Token 分配给第i个专家的“置信度权重”.
# Top-k策略：在TC模式下，每个token选择最适合处理的专家；在EC模式下，每个专家主动选择最适合处理的token

class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.linear = Linear(d_model,num_experts)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return logits

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.experts = nn.ModuleList([
        #     SwiGLU(d_model,d_ff) for _ in range(num_experts)
        #      ])
        self.expert = SwiGLU(d_model,d_ff)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)
    
class MoE(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_experts: int,
            top_k: int = 2,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.router = Router(d_model,num_experts)
        self.experts = nn.ModuleList([
            SwiGLU(d_model,d_ff) for _ in range(num_experts)
             ])


    def forward(self,x:torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        # 将序列展平，把所有的 Token 视为独立个体: [N, d_model], N = batch_size * seq_len
        x_flat = x.view(-1,d_model)
        N = x_flat.shape[0]

        # 计算 Router Logits 与 Z-Loss
        router_logits = self.router(x_flat)

        z_loss = torch.logsumexp(router_logits,dim=-1).pow(2).mean()
        # 计算路由概率与 Aux Loss (负载均衡)

        routing_probs = softmax(router_logits,dim = -1)
        # 取出最高概率的专家索引
        topk_probs, topk_indices = torch.topk(routing_probs,self.top_k,dim = -1)
        # 统计每个专家的实际 Token 分配比例 (f_i)
        expert_mask = torch.zeros_like(routing_probs).scatter_(1, topk_indices, 1.0)
        tokens_per_expert = expert_mask.sum(dim=0) # [num_experts]
        f_i = tokens_per_expert / (N * self.top_k)
        # 统计每个专家的平均期望概率 (P_i)
        p_i = routing_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f_i * p_i)

        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output_flat = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            # where返回所有满足条件的位置坐标,[x,]
            token_indices, k_idx = torch.where(topk_indices == i)

            if token_indices.shape[0] == 0:
                continue
            # 把这些被选中的 Token 物理抽取出来: [num_tokens_in_expert_i, d_model]
            dispatched_tokens = x_flat[token_indices]
            # 分别过对应专家的ffn
            # [N,d_model]
            expert_output = expert(dispatched_tokens)
            # [N,topk]->[num_tokens_in_expert_i, ]->[num_tokens_in_expert_i, 1]
            expert_weights = topk_probs[token_indices, k_idx].unsqueeze(-1)

            output_flat[token_indices] += expert_output * expert_weights

        total_layer_aux_loss = 0.01 * aux_loss + 1e-4 * z_loss

        output = output_flat.view(batch_size, seq_len, d_model)

        return output, total_layer_aux_loss



        






