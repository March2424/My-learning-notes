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
    def __init__(self, d_model: int, num_experts: int, topk: int = 2):
        super().__init__()
        self.d_model = d_model
        self.topk = topk
        self.num_experts = num_experts
        self.gate = Linear(d_model,num_experts)


    def forward(self,x : torch.Tensor):
        logits = self.gate(x)
        logits_topk, logits_index = logits.topk(self.topk, dim = -1)
        #创建一个权威负无穷矩阵，为了之后经过softmax之后，无关的expert被设置为0
        zeros = torch.full_like(logits,float("-inf"))
        sparse_logits = zeros.scatter(-1,logits_index,logits_topk)
        sparse_logits = softmax(sparse_logits,-1)
        # PyTorch 中计算交叉熵或辅助损失的函数，通常期望输入的特征维度是二维的 
        # 将其展平为 (b * s, expert_num) 可以直接无缝喂给损失函数计算公式。
        # 计算auxloss时候，不关心token属于哪个batch或是句子中的哪个位置。只关心一共多少个token分给了专家a或b。。
        gate_logits = logits.view(-1,self.num_experts)

        return sparse_logits,logits_index,gate_logits

class Experts(nn.Module):
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
            z_loss_coef: float = 1e-3,
            lb_loss_coef: float = 1e-1,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.router = Router(d_model,num_experts,top_k)
        self.z_loss_coef = z_loss_coef
        self.lb_loss_coef = lb_loss_coef
        self.experts = nn.ModuleList([
            SwiGLU(d_model,d_ff) for _ in range(num_experts)
             ])
    
    def forward(self,x: torch.Tensor):
        sparse_logits, logits_index, gate_logits = self.router(x)
        b,s,d_model = x.shape
        x_flat = x.view(-1,d_model)
        sparse_logits = sparse_logits.view(-1,self.num_experts)
        logits_index = logits_index.view(-1, self.top_k)
        # x_flat[b*s,d_model] logits_index[b*s,top_k]
        final_output = torch.zeros_like(x_flat)
        z_loss = self._z_loss(gate_logits)
        for expert_idx in range(self.num_experts):
            # (b*S,)
            token_mask = (logits_index == expert_idx).any(dim = -1)
            if not token_mask.any():
                continue
            # selected_x[M,d_model]
            selected_x = x_flat[token_mask]
            # [M,d_model]
            expert_output = self.experts[expert_idx](selected_x)
            # weighted_outputs = expert_outputs * gating_weights [M,d_model]
            # gating_weights[M,1]
            gating_weights = sparse_logits[token_mask,expert_idx].unsqueeze(-1)
            weighted_output = expert_output * gating_weights
            # 最后要根据top-k计算出的概率来加权最后的结果
            final_output[token_mask] += weighted_output
            
        full_router_probs = softmax(gate_logits, dim=-1)
            
        total_layer_aux_loss = self.lb_loss_coef * self._load_balance_loss(full_router_probs,logits_index, self.num_experts) + self.z_loss_coef * z_loss

        return final_output.view(b,s,d_model), total_layer_aux_loss
    
    @staticmethod
    def _load_balance_loss(
        router_probs: torch.Tensor,  # (B, S, expert_num) softmax(logits)
        topk_indices: torch.Tensor,  # (B, S, topk)
        num_experts: int,
    ) -> torch.Tensor:
        p_i = router_probs.mean(dim=(0,1))
        dispatch = F.one_hot(topk_indices, num_classes=num_experts).to(router_probs.dtype)
        f_i= dispatch.mean(dim = (0,1,2))
        return num_experts * torch.sum(p_i*f_i)
    
    @staticmethod
    def _z_loss(logits: torch.Tensor) -> torch.Tensor:
        log_sum_exp = torch.logsumexp(logits,dim = -1)
        # [b*s,num_experts]
        z_loss = torch.mean(log_sum_exp**2)
        return z_loss

        

