import calc_math
import torch
import torch.nn as nn
import math


class AKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model=256,
        n_blocks=1,
        dropout=0.1,
        kq_same=1,
        model_type='akt',
        l2=1e-5
    ):
        super(AKT, self).__init__()

        self.n_question = n_question
        self.n_pid = n_pid
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.kq_same = kq_same

        # ===== Embedding =====
        # padding_idx=0：0 作为 padding，不参与训练更新，避免 padding 污染表示
        self.q_embed = nn.Embedding(n_question + 1, d_model, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * n_question + 1, d_model, padding_idx=0)

        # PID difficulty embedding
        self.difficult_param = nn.Embedding(n_pid + 1, d_model, padding_idx=0)

        # ===== Transformer Blocks =====
        self.blocks = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                d_ff=d_model * 4,
                n_heads=8,
                dropout=dropout,
                kq_same=kq_same
            )
            for _ in range(n_blocks)
        ])

        # ===== Output =====
        self.out = nn.Linear(d_model, 1)

    def forward(self, q_data, qa_data, pid_data):
        """
        q_data:  [batch, seq]
        qa_data: [batch, seq]  (qa = q + a*n_question)
        pid_data:[batch, seq]
        """

        # ===== Embedding =====
        q_embed = self.q_embed(q_data)          # [B,S,D]
        qa_embed = self.qa_embed(qa_data)       # [B,S,D]
        pid_embed = self.difficult_param(pid_data)

        # =========================================================
        # 关键修复：把 qa_embed 右移一位（shift）
        # 让位置 t 的预测只能看到 (t-1) 及以前的作答信息
        # 避免训练阶段“看见同位置答案”造成泄漏
        # =========================================================
        qa_shift = torch.zeros_like(qa_embed)
        qa_shift[:, 1:, :] = qa_embed[:, :-1, :]   # t位置用到的是(t-1)的qa
        # qa_shift[:,0,:] 保持为0（无历史）

        # AKT 核心：题目 embedding + 难度参数
        x = q_embed + pid_embed

        # ===== Transformer =====
        for block in self.blocks:
            x = block(x, qa_shift)

        # ===== Prediction =====
        out = torch.sigmoid(self.out(x)).squeeze(-1)  # [B,S]
        return out


# =========================================================
# Transformer Layer
# =========================================================

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kq_same=kq_same
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        # Self-Attention（q来自x，k/v来自y=qa_shift）
        attn_out = self.attn(x, y, y)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# =========================================================
# Multi-Head Attention
# =========================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.kq_same = kq_same

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size, seq_len, _ = q.size()

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 仍然使用因果mask（可看自己与过去）
        # 因为我们已经做了 qa_shift，自身位置不再含“当前答案”，不会泄漏
        causal = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).bool()
        scores = scores.masked_fill(~causal, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)
