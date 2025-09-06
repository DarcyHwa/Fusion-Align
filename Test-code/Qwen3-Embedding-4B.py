# Requires: transformers>=4.51.0
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

assert torch.cuda.is_available(), "需要可用的 NVIDIA GPU（CUDA）。请在有 CUDA 的环境下运行。"

MODEL_ID   = "Qwen/Qwen3-Embedding-4B"
MAX_LENGTH = 256  # 常规句子足够

# 若只想预览前 N 维，设置为整数；打印完整向量则设为 None
PREVIEW_DIMS = 5  # 例如 10

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """取每个样本最后一个有效 token（兼容左/右 padding）。"""
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:  # 左填充
        return last_hidden_states[:, -1]
    idx = attention_mask.sum(dim=1) - 1
    batch = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
    return last_hidden_states[batch, idx]

# --- 分词器 ---
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left", use_fast=True)
# 安全兜底：若无 pad_token，则用 eos_token 充当
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

# --- 仅 GPU：优先 BF16，否则 FP16 ---
dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = torch.device("cuda")
model  = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
BACKEND = "GPU-BF16" if dtype == torch.bfloat16 else "GPU-FP16"
print("Backend =", BACKEND, "| dtype =", dtype)

@torch.inference_mode()
def embed_texts(texts):
    """
    输入：str 或 List[str]
    输出：(N, hidden_size) 的 L2 归一化句向量（点积≈余弦）
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    batch = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        pad_to_multiple_of=8,   # 对 Tensor Cores 友好
        return_tensors="pt",
    )
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    out  = model(**batch, return_dict=True)                 # 不要 output_hidden_states
    sent = last_token_pool(out.last_hidden_state, batch["attention_mask"])
    sent = F.normalize(sent, p=2, dim=1)
    return (texts[0], sent[0]) if single else (texts, sent)

def print_embeddings(texts, embeddings):
    """逐条打印原句与对应的句向量。"""
    if isinstance(texts, str):
        texts = [texts]
        embeddings = embeddings.unsqueeze(0)

    for i, t in enumerate(texts):
        vec = embeddings[i].detach().cpu().tolist()
        if PREVIEW_DIMS is not None:
            vec_print = vec[:PREVIEW_DIMS]
            tail_note = f" ... (total {len(vec)} dims)"
        else:
            vec_print = vec
            tail_note = ""
        print(f"\n文本：{t}")
        print(f"向量维度：{len(vec)}")
        print(f"嵌入向量：{vec_print}{tail_note}")

# ===== 用法示例 =====
if __name__ == "__main__":
    # 单句
    s_text, s_emb = embed_texts("采用优等生鲜肉,欢迎新老师生前来就餐。")
    print_embeddings(s_text, s_emb)

    # 句子列表
    arr_texts, arr_embs = embed_texts(["北京是中国的首都。", "欢迎光临本店。", "Gravity gives weight to objects."])
    print_embeddings(arr_texts, arr_embs)

    # 可选：相似度矩阵（点积=余弦）
    sims = (arr_embs @ arr_embs.T).cpu().tolist()
    print("\n相似度矩阵：\n", sims)
