#!/usr/bin/env python3
import torch
import torch.nn as nn

class ExplainableDecoderLayer(nn.Module):
    """
    一个 "可解释" 的 Transformer Decoder Layer。
    功能与 nn.TransformerDecoderLayer 一致，但支持返回 Cross-Attention 权重。
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 1. Self-Attention (Emotion Queries 之间的交互)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Cross-Attention (Emotion Query -> Multimodal Memory)
        # 这就是要可视化的核心部分！
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

    def forward(
        self, 
        tgt, 
        memory, 
        memory_key_padding_mask=None, 
        return_attention=False
    ):
        # --- 1. Self Attention ---
        # query=tgt, key=tgt, value=tgt
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # --- 2. Cross Attention ---
        # query=tgt (Emotion), key=memory (Fused Seq), value=memory
        # 这里的 weights 就是 "Emotion-Level Interpretability"
        tgt2, cross_attn_weights = self.cross_attn(
            query=tgt, 
            key=memory, 
            value=memory, 
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attention # <--- 关键开关
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # --- 3. FFN ---
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))

        if return_attention:
            return tgt, cross_attn_weights
        else:
            return tgt, None

class EmotionDecoder(nn.Module):
    """
    Emotion-Level Transformer Decoder with Learnable Emotion Queries.

    Each emotion is represented by a learnable query vector.
    The decoder uses cross-attention from these queries to the fused
    multimodal sequence (memory) to extract emotion-specific evidence.

    Inputs:
        memory: [B, L, d]  - fused sequence from cross-modal fusion
        memory_key_padding_mask: [B, L] bool or None (True = PAD)

    Outputs:
        z:      [B, num_emotions, d] - emotion-specific embeddings
        logits: [B, num_emotions] or None (if classifier provided outside)

    支持 return_attention=True 用于可解释性分析。
    """

    def __init__(
        self,
        d_model: int = 768,
        num_emotions: int = 4,
        n_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_output_layer: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_emotions = num_emotions
        self.use_output_layer = use_output_layer

        # Learnable emotion queries: [num_emotions, d]
        self.emotion_queries = nn.Parameter(
            torch.randn(num_emotions, d_model)
        )

        # 替换原本的 nn.TransformerDecoder，改用我们要手写的 ModuleList
        self.layers = nn.ModuleList([
            ExplainableDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Optional output projection
        if use_output_layer:
            self.out_proj = nn.Linear(d_model, 1)
        else:
            self.out_proj = None

    def forward(
        self,
        memory: torch.Tensor,                      # [B, L, d]
        memory_key_padding_mask: torch.Tensor | None = None,  # [B, L] or None
        return_attention: bool = False             # <--- 新增接口
    ):
        B = memory.size(0)

        # 1) Expand emotion queries for the batch
        #    queries: [B, num_emotions, d]
        queries = self.emotion_queries.unsqueeze(0).expand(B, -1, -1)

        # 2) Run Manual Decoder Loop
        #    - queries attend over "memory" (fused multimodal sequence)
        #    - no causal mask (we want full context)
        all_layers_attn = [] # 用来存每一层的 attention

        out = queries
        for layer in self.layers:
            # 每一层都尝试获取 attention
            out, attn_map = layer(
                tgt=out, 
                memory=memory, 
                memory_key_padding_mask=memory_key_padding_mask,
                return_attention=return_attention
            )
            
            if return_attention and attn_map is not None:
                # attn_map shape: [B, Num_Emotions, Seq_Len]
                all_layers_attn.append(attn_map)

        # 3) Output Projection
        z = out # [B, num_emotions, d]

        logits = None
        # Optionally map each emotion embedding to a logit
        if self.out_proj is not None:
            # out_proj per emotion: [B, num_emotions, 1] -> [B, num_emotions]
            logits = self.out_proj(z).squeeze(-1)

        # 4) Return logic
        if return_attention:
            # 返回: z, logits, 以及所有层的 attention 列表
            return z, logits, all_layers_attn
        else:
            return z, logits
