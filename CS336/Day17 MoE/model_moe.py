import torch
import torch.nn as nn
import math
from einops import rearrange
from MoE import MoE

class Linear(nn.Module):
    def __init__(self, in_features: int,
                out_features: int,
                device: torch.device | None=None,
                dtype: torch.dtype | None = None):
        # 继承父类
        super().__init__()
        kwargs = {'device': device,'dtype':dtype}
        self.in_features = in_features
        self.out_features = out_features
        # 先分配内存，因为nn.parameter必须包装一个已经存在的tensor,empty只要求一块内存不写入,不使用bias
        self.weight = nn.Parameter(torch.empty((out_features,in_features),**kwargs))

        std = math.sqrt(2.0 / (self.in_features+self.out_features))
        nn.init.trunc_normal_(self.weight,mean = 0.0,std = std,a=-3*std,b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[batch，seq,in] @ weight[out,in]T
        return torch.einsum('...i, oi -> ...o',x,self.weight)

class embedding(nn.Module):
    # num_embeddings: int为vocab_size embedding_dim: int是嵌入向量的维度即d_model
    def __init__(self, num_embeddings: int, embedding_dim: int,
                device: torch.device | None=None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        kwargs = {'device': device,'dtype':dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**kwargs))
        nn.init.trunc_normal_(self.weight,mean = 0.0, std = 1,a=-3.0,b=3.0)
        

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_id[batch_size,seq_len]
        # 从weight[vocab_size,d_model]中lookup， 输出[batch_size,seq,d_model]
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        kwargs = {'device': device,'dtype':dtype}
        self.weight = nn.Parameter(torch.ones(d_model,**kwargs))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[batch_size,seq_len,d_model] * [d_model,1]逐元素相乘
        '''
        关于并行处理的想法：让batchsize的句子里的所有seqlen大小的词乘以了self.weight
        算出一个总的loss，在前向传播中weight被广播到了[batch_size,seq_len,d_model]的形状
        所以最终需要把所有分支梯度求和，最终self.weight.grad仍然为[d_model] 
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        rms_result = x / result
        return (rms_result * self.weight).to(in_dtype)

def SiLU(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(in_features=d_model,out_features=d_ff)
        self.w2 = Linear(in_features=d_ff,out_features=d_model)
        self.w3 = Linear(in_features=d_model,out_features=d_ff)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # [batchsize,seq_len,d_model] -> [...,d_model]
        return self.w2(SiLU(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    '''
    r(n)的行乘以q，v矩阵的行
    rope的同一个token有d_head维，被划分为k=dh/2块，每一块的旋转角度根据公式不同分配不同的转速。
    这样，每一个绝对位置i，在这d/2个平面的组合相位都是唯一的。没有任何两个 Token 会发生重合。
    保证注意力机制的长短衰减
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        powers = torch.arange(0,d_k,2,device=device).float() / d_k
        # [d_k/2,]
        freqs = 1.0 / (theta ** powers)
        # [max_seq_len,]
        t = torch.arange(max_seq_len,device=device).float()
        freqs_martrix = torch.outer(t,freqs) #[max_len,d_k/2]
        self.register_buffer("cos_cached", freqs_martrix.cos(),persistent=False)
        self.register_buffer("sin_cached", freqs_martrix.sin(),persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # [..., seq_len, dim], [seq_len,] -> [..., seq_len, dim]
        # 提取[...,seq,d_k/2]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        # 将d_k 拆解为 (d_k/2, 2)，分别分给x1，x2
        # [...,seq_len,d_k/2]
        x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        # x1[x1,x3,x5,...] x2[x2,x4,x6,...] x_out[...,seq_len,d_k/2,2]
        x_out = torch.stack([x1*cos-x2*sin,
                             x2*cos+x1*sin],dim = -1)
        # [..., seq_len, d_k]
        x_out = x_out.flatten(-2,-1)

        return x_out
    
def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits-max_logits)
    sum_exp = torch.sum(exp_logits,dim=dim,keepdim=True)
    res_softmax = exp_logits / sum_exp
    return res_softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,        
    mask: torch.Tensor = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0,float('-inf'))
    attn_weights = softmax(scores,dim=-1)
    output = torch.matmul(attn_weights,V)
    return output
    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                max_seq_len: int = 2048,
                theta: float = 10000.0,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
                ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.ln_q = RMSNorm(self.head_dim,device=device,dtype=dtype)
        self.ln_k = RMSNorm(self.head_dim,device=device,dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.head_dim,max_seq_len,device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_position: torch.Tensor = None) -> torch.Tensor:
        batch_size,seq_len,d_model = x.shape
        # [batchsize,seqlen,dmodel] -> [batchsize,nhead,seq,head_dim]
        q_head = self.q_proj(x).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        k_head = self.k_proj(x).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        v_head = self.v_proj(x).view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)

        q_head = self.ln_q(q_head)
        k_head = self.ln_k(k_head)


        if self.rope:
            if token_position is not None:
                q_head,k_head = self.rope.forward(q_head,token_position),self.rope.forward(k_head,token_position)
        
        mask = torch.tril(torch.ones(seq_len,seq_len,device=x.device,dtype=torch.bool))
        
        # (batch_size,num_head,seq_len,head_dim)
        attn_output = scaled_dot_product_attention(q_head,k_head,v_head,mask)

        attn_output = rearrange(attn_output,'... h s d -> ... s (h d)')

        return self.output_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.mha = MultiheadSelfAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        # self.ffn = SwiGLU(d_model,d_ff)
        self.moe = MoE(d_model, d_ff, num_experts=4, top_k=1)
        #实现Initialization of projections to zero
        self.mha.output_proj.is_residual_output = True
        # self.ffn.w2.is_residual_output = True
        for expert in self.moe.experts:
            expert.w2.is_residual_output = True
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        
    
    # 实现的是prenorm，可以尝试一下qk norm以及hybird norm（对于小模型效果不佳）
    def forward(self,x: torch.Tensor, token_positions: torch.Tensor = None):

        x = x + self.mha(self.ln1(x),token_positions)

        # 接收 MoE 的输出和 auxiliary loss
        moe_out, layer_aux_loss = self.moe(self.ln2(x))
        x = x + moe_out

        return x,layer_aux_loss


# class output_layer(nn.Module):
#     def __init__(self, d_model: int, vocab_size:int):
#         super().__init__()
#         self.linear = Linear(d_model,vocab_size)
#         self.norm = RMSNorm(d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.norm(x)
#         logits = self.linear(x)
#         return logits
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        context_length:int,
        d_model:int,
        num_layers:int,
        num_heads:int,
        d_ff:int,
        rope_theta:float,
        device=None,dtype = None,
        use_rms_norm: bool = True,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu"
    ):
        super().__init__()
        
        self.token_embeddings = embedding(vocab_size,d_model,device=device,dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,num_heads,d_ff,context_length,rope_theta,
                device=device,dtype=dtype
            )
            for _ in range(num_layers)
        ])


        if use_rms_norm:
            self.final_norm = RMSNorm(d_model,device = device,dtype=dtype)
        else:
            self.final_norm = nn.Identity()

        self.linear_out = Linear(d_model,vocab_size)

        # 添加权重绑定(Weight Tying)
        #令linear_oout的权重矩阵 指向token_embedding的权重矩阵
        self.linear_out.weight = self.token_embeddings.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, Linear):
            if hasattr(m, 'is_residual_output') and m.is_residual_output:
                torch.nn.init.zeros_(m.W) #核心投影层权重直接归零
                
            else:
                std = math.sqrt(2.0 / (m.in_features + m.out_features))
                nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                
        elif isinstance(m, embedding):
            #针对绑定后的Embedding进行初始化
            std = math.sqrt(2.0 / (m.in_features + m.out_features))
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # [batch_size,seq_len] -> [batch_size,seq_len,d_model]
        b,s = token_ids.shape
        token_positions = torch.arange(s,device=token_ids.device)
        x = self.token_embeddings(token_ids)

        total_aux_loss = 0.0

        for layer in self.layers:
            x,layer_aux_loss = layer(x,token_positions)
            total_aux_loss+=layer_aux_loss
        


        x = self.final_norm(x)

        logits = self.linear_out(x)

        return logits, total_aux_loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids:torch.Tensor,
        max_new_tokens:int, 
        eos_token_id:int = None,
        temperature:float = 1.0,
        top_p:float = 1.0,
    ) -> torch.Tensor:
        self.eval()

        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.context_length:]
            # [batch_size,seq_len,vocab] -> [batch_size,1,vocab]
            logit = self.forward(idx_cond)
            logit = logit[:,-1,:]

            if temperature != 0:
                logits = logits / (temperature + 1e-8)

            if top_p < 1.0:
                logits = self.top_p_filter(logits,top_p)

            probs = softmax(logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1) # (Batch, 1)

            generated = torch.cat((generated,next_token),dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
    
    def top_p_filter(
            self,
            logits:torch.Tensor,
            top_p: float
    ) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(logits, descending=True,dim=-1) # from high to low
        cumsum_probs = torch.cumsum(softmax(sorted_probs, dim=-1), dim=-1)
        # Find a smallest set, let its sum >= top_p
        # cumsum_probs - sorted_probs means the all-left-probs sum
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits









    