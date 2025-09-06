#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 neavo/modern_bert_multilingual 模型 获取子词级向量。
功能：
- 对句子列表中每个句子进行分词
- 显示分词结果和对应的分词向量信息
- 计算双语句子的词级别对齐
运行环境:
  pip install transformers torch numpy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import unicodedata

# 固定的编码层号常量
LAYER_NUM = 8

print(torch.cuda.is_available())      # 应返回 True
print(torch.version.cuda)             # 应显示 CUDA 版本号

print("\n" + "="*60)
# 检测与显示设备信息
if torch.cuda.is_available():
    device_str = f"cuda:{torch.cuda.current_device()}"
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"[Device] Using GPU: {device_str} - {gpu_name}")
    device = torch.device(device_str)
else:
    device_str = "cpu"
    print(f"[Device] Using CPU")
    device = torch.device("cpu")


print("\n" + "="*60)
print("模型加载")
print("="*60)

model_id = "facebook/xlm-roberta-xl"

# 加载 tokenizer 和模型
print(f"[模型信息] 正在加载模型: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, add_pooling_layer=False)
model = model.to(device)
model.eval()

# 获取模型配置信息
if hasattr(model, 'config'):
    print(f"[模型信息] 总层数: {getattr(model.config, 'num_hidden_layers', 'N/A')}")
    print(f"[模型信息] 隐藏层维度: {getattr(model.config, 'hidden_size', 'N/A')}")

print("\n" + "="*60)


def get_layer_embeddings(model, tokenizer, sentences):
    """
    获取指定层（常量 LAYER_NUM）的词嵌入向量
    
    Args:
        model: 模型实例
        tokenizer: 对应的 tokenizer
        sentences: 句子列表或单个句子字符串
    
    Returns:
        embeddings: 词嵌入向量列表
        tokens_list: 分词结果列表
    """
    # 确保输入是列表格式
    if isinstance(sentences, str):
        sentences = [sentences]
    
    embeddings = []
    tokens_list = []
    
    for sentence in sentences:
        # 分词
        inputs = tokenizer(sentence, return_tensors="pt", padding=False, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取分词结果
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens_list.append(tokens)
        
        # 获取模型输出，包括所有隐藏层
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # 提取常量层的输出（不做越界检查，遵循你的偏好）
        # hidden_states[0] 为输入嵌入，第 1..N 为第 1..N 层输出
        layer_output = outputs.hidden_states[LAYER_NUM]  # shape: [batch_size, seq_len, hidden_size]
        embeddings.append(layer_output[0])  # 取第一个（也是唯一的）batch
    
    return embeddings, tokens_list


def display_tokenization_and_vectors(model, tokenizer, sentences, lang_label=""):
    """
    对句子列表进行分词并显示分词结果和向量信息
    
    Args:
        model: 模型实例
        tokenizer: 对应的 tokenizer
        sentences: 句子列表或单个句子字符串
        lang_label: 语言标签（用于显示）
    
    Returns:
        embeddings: 词嵌入向量列表
        tokens_list: 分词结果列表
    """
    # 确保输入是列表格式
    if isinstance(sentences, str):
        sentences = [sentences]
    
    print(f"\n{'='*60}")
    print(f"处理{lang_label}句子列表 (使用第{LAYER_NUM}层):")
    print(f"{'='*60}")
    
    # 获取词嵌入和分词结果
    embeddings, tokens_list = get_layer_embeddings(model, tokenizer, sentences)
    
    # --- 工具函数：解码与判断 ---
    def decode_token(token: str) -> str:
        """将单个 token 解码为可读文本（去除多余空白）。"""
        try:
            s = tokenizer.convert_tokens_to_string([token]).strip()
            return s if s != "" else token
        except Exception:
            return token

    def is_special_token(token: str) -> bool:
        """是否为分词器生成的特殊标记，如 <s>、</s>、<pad> 等。"""
        if token in getattr(tokenizer, 'all_special_tokens', []):
            return True
        if token.startswith('<') and token.endswith('>'):
            return True
        # 若解码为空字符串，则视为无用标记
        try:
            if tokenizer.convert_tokens_to_string([token]).strip() == "":
                return True
        except Exception:
            pass
        return False

    def is_punct_or_symbol_text(text: str) -> bool:
        """判断一个解码后的文本是否全为标点/符号。"""
        if not text:
            return False
        for ch in text:
            cat = unicodedata.category(ch)
            if not (cat.startswith('P') or cat.startswith('S')):
                return False
        return True

    def aggregate_english_tokens_to_words(tokens: list, embedding_tensor: torch.Tensor):
        """
        将英文 SPM 子词聚合为“词”级别，并保留标点为独立单元；
        返回：words(list[str]), spans(list[list[int]]), pooled_vecs(torch.Tensor[num_words, hidden])
        """
        spans = []
        words = []
        current_span = []

        def flush_span():
            nonlocal current_span
            if current_span:
                spans.append(current_span)
                # 以 span 内 token 还原文本（用于显示）
                token_text = tokenizer.convert_tokens_to_string([tokens[i] for i in current_span]).strip()
                words.append(token_text if token_text else ''.join([decode_token(tokens[i]) for i in current_span]))
                current_span = []

        for i, tok in enumerate(tokens):
            if is_special_token(tok):
                flush_span()
                continue
            text = decode_token(tok)
            # 遇到纯标点：作为独立单元
            if is_punct_or_symbol_text(text):
                flush_span()
                spans.append([i])
                words.append(text)
                continue

            if tok.startswith('▁'):
                # 新词开始
                flush_span()
                current_span = [i]
            else:
                if not current_span:
                    current_span = [i]
                else:
                    current_span.append(i)

        flush_span()

        # 平均池化
        if len(spans) == 0:
            pooled = torch.empty((0, embedding_tensor.shape[-1]), device=embedding_tensor.device)
        else:
            pooled = []
            for span in spans:
                valid = [j for j in span if j < embedding_tensor.shape[0]]
                if not valid:
                    pooled.append(torch.zeros((embedding_tensor.shape[-1],), device=embedding_tensor.device))
                else:
                    pooled.append(embedding_tensor[valid].mean(dim=0))
            pooled = torch.stack(pooled, dim=0)
        return words, spans, pooled

    def filter_tokens_keep_punct(tokens: list, embedding_tensor: torch.Tensor):
        """
        中文路径：仅移除分词器特殊标记，保留原始标点；
        返回：texts(list[str]), kept_indices(list[int]), kept_vecs(torch.Tensor[num, hidden])
        """
        kept_indices = [i for i, tok in enumerate(tokens) if not is_special_token(tok)]
        texts = [decode_token(tokens[i]) for i in kept_indices]
        if len(kept_indices) == 0:
            kept_vecs = torch.empty((0, embedding_tensor.shape[-1]), device=embedding_tensor.device)
        else:
            kept_vecs = embedding_tensor[kept_indices]
        return texts, kept_indices, kept_vecs

    # 对每个句子，显示分词结果和向量信息（原始 + 清理/合并后）
    for idx, (sentence, tokens, embedding) in enumerate(zip(sentences, tokens_list, embeddings)):
        print(f"\n句子 {idx + 1}: {sentence}")
        print("-" * 70)
        
        # 显示基本信息
        print(f"分词数量: {len(tokens)}")
        print(f"向量维度: {embedding.shape} (序列长度: {embedding.shape[0]}, 向量维度: {embedding.shape[1]})")
        
        # 1) 原始分词结果和对应的向量值
        print(f"\n[原始] 分词结果及前10个向量值:")
        print("-" * 70)
        
        for i in range(min(len(tokens), embedding.shape[0])):
            token = tokens[i]
            # 处理特殊字符显示，统一用解码后的结果；若解码为空则回退原 token
            display_token = decode_token(token)
            
            # 限制显示长度
            if len(display_token) > 15:
                display_token = display_token[:12] + '...'
            
            # 获取 token ID（可能为 -1 表示未知）
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is None:
                    token_id = -1
            except Exception:
                token_id = -1
            
            # 获取前10个向量值
            vec_values = embedding[i][:10]
            
            # 格式化显示，确保对齐
            print(f"  [{i:2d}] {display_token:<15s} (ID: {token_id:6d}) -> ", end="")
            
            # 格式化向量值
            if isinstance(vec_values, torch.Tensor):
                values_str = ", ".join([f"{v:7.4f}" for v in vec_values.cpu().tolist()])
            else:
                values_str = ", ".join([f"{v:7.4f}" for v in vec_values])
            
            print(f"[{values_str}]")

        # 2) 清理无关标记后视图：英文合并为词（平均池化），中文仅清理不合并，均保留原始标点
        print(f"\n[清理后] (仅移除分词器特殊标记；英文合并为词，中文不合并，均保留标点)")
        print("-" * 70)
        if '英' in lang_label:
            words, spans, pooled_vecs = aggregate_english_tokens_to_words(tokens, embedding)
            print(f"合并后单元数量: {len(words)}")
            if pooled_vecs.shape[0] > 0:
                print(f"向量维度: {pooled_vecs.shape} (单元数: {pooled_vecs.shape[0]}, 向量维度: {pooled_vecs.shape[1]})")
            for i, w in enumerate(words):
                vec_head = pooled_vecs[i][:10] if pooled_vecs.shape[0] > 0 else torch.empty(0)
                if isinstance(vec_head, torch.Tensor):
                    values_str = ", ".join([f"{v:7.4f}" for v in vec_head.detach().cpu().tolist()])
                else:
                    values_str = ""
                disp = w
                if len(disp) > 20:
                    disp = disp[:18] + '..'
                print(f"  [{i:2d}] {disp:<20s} -> [{values_str}]")
        else:
            texts, kept_indices, kept_vecs = filter_tokens_keep_punct(tokens, embedding)
            print(f"清理后 token 数量: {len(texts)}")
            if kept_vecs.shape[0] > 0:
                print(f"向量维度: {kept_vecs.shape} (序列长度: {kept_vecs.shape[0]}, 向量维度: {kept_vecs.shape[1]})")
            for i, t in enumerate(texts):
                vec_head = kept_vecs[i][:10] if kept_vecs.shape[0] > 0 else torch.empty(0)
                if isinstance(vec_head, torch.Tensor):
                    values_str = ", ".join([f"{v:7.4f}" for v in vec_head.detach().cpu().tolist()])
                else:
                    values_str = ""
                disp = t
                if len(disp) > 15:
                    disp = disp[:13] + '..'
                print(f"  [{i:2d}] {disp:<15s} -> [{values_str}]")
    
    return embeddings, tokens_list


def compute_word_alignment(embeddings_en, embeddings_zh, tokens_en, tokens_zh):
    """
    计算中英文句子的词级别对齐和相似度（英文侧按“单词”聚合）。
    
    逻辑：
    - 英文侧：SentencePiece 子词分词结果中，以 '▁' 开头的 token 作为新词边界，将连续子词聚合为单词；
      对每个英文单词的子词向量做平均池化，得到“词级向量”。
    - 中文侧：保持原有子词/字级向量不变。
    - 相似度矩阵：形状为 英文词 × 中文 token。
    
    Args:
        embeddings_en: 英文句子的子词级嵌入 (seq_len, hidden)
        embeddings_zh: 中文句子的子词级嵌入 (seq_len, hidden)
        tokens_en: 英文分词结果（子词列表，或 [子词列表] 的嵌套）
        tokens_zh: 中文分词结果（子词列表，或 [子词列表] 的嵌套）
    
    Returns:
        alignment_matrix: 词对齐相似度矩阵（英文词 × 中文 token）
        words_en: 英文单词列表（与矩阵行对应）
        tokens_zh: 中文子词列表（与矩阵列对应）
    """
    # 获取子词级向量
    vec_en = embeddings_en[0] if isinstance(embeddings_en, list) else embeddings_en
    vec_zh = embeddings_zh[0] if isinstance(embeddings_zh, list) else embeddings_zh

    # 转换为 numpy 数组
    if isinstance(vec_en, torch.Tensor):
        vec_en = vec_en.cpu().numpy()
    if isinstance(vec_zh, torch.Tensor):
        vec_zh = vec_zh.cpu().numpy()

    # 展平 tokens 列表
    tokens_en_sub = tokens_en[0] if isinstance(tokens_en[0], list) else tokens_en
    tokens_zh = tokens_zh[0] if isinstance(tokens_zh[0], list) else tokens_zh

    # 解码函数
    def decode_token(token: str) -> str:
        try:
            s = tokenizer.convert_tokens_to_string([token]).strip()
            return s if s != "" else token
        except Exception:
            return token

    def is_special_token(token: str) -> bool:
        if token in getattr(tokenizer, 'all_special_tokens', []):
            return True
        if token.startswith('<') and token.endswith('>'):
            return True
        try:
            if tokenizer.convert_tokens_to_string([token]).strip() == "":
                return True
        except Exception:
            pass
        return False

    def is_punct_text(text: str) -> bool:
        if not text:
            return False
        for ch in text:
            cat = unicodedata.category(ch)
            if not (cat.startswith('P') or cat.startswith('S')):
                return False
        return True

    # 英文：聚合为词并保留标点为独立单元
    words_en = []
    word_spans_en = []
    current_span = []

    def flush_span():
        nonlocal current_span
        if current_span:
            word_spans_en.append(current_span)
            token_text = tokenizer.convert_tokens_to_string([tokens_en_sub[i] for i in current_span]).strip()
            words_en.append(token_text if token_text else ''.join([decode_token(tokens_en_sub[i]) for i in current_span]))
            current_span = []

    for idx, tok in enumerate(tokens_en_sub):
        if is_special_token(tok):
            flush_span()
            continue
        text = decode_token(tok)
        if is_punct_text(text):
            flush_span()
            word_spans_en.append([idx])
            words_en.append(text)
            continue
        if tok.startswith('▁'):
            flush_span()
            current_span = [idx]
        else:
            if not current_span:
                current_span = [idx]
            else:
                current_span.append(idx)
    flush_span()

    # 将英文子词向量聚合为词级向量（平均池化）
    en_word_vecs = []
    hidden_size = vec_en.shape[1] if vec_en.ndim == 2 else (vec_zh.shape[1] if vec_zh.ndim == 2 else 0)
    for span in word_spans_en:
        valid_indices = [i for i in span if i < vec_en.shape[0]]
        if not valid_indices:
            en_word_vecs.append(np.zeros(hidden_size, dtype=np.float32))
            continue
        vecs = vec_en[valid_indices]
        en_word_vecs.append(vecs.mean(axis=0))

    # 中文：仅移除特殊标记，保留原标点
    zh_keep_indices = [j for j, t in enumerate(tokens_zh) if not is_special_token(str(t))]
    tokens_zh_filtered = [decode_token(tokens_zh[j]) for j in zh_keep_indices]
    zh_vecs = vec_zh[zh_keep_indices] if len(zh_keep_indices) > 0 else np.zeros((0, vec_zh.shape[1]))

    # 计算“英文词 × 中文 token”的相似度矩阵（先进行 L2 归一化，再用点积，等价于余弦相似度）
    n_tokens_en = len(en_word_vecs)
    n_tokens_zh = zh_vecs.shape[0]
    if n_tokens_en == 0 or n_tokens_zh == 0:
        alignment_matrix = np.zeros((n_tokens_en, n_tokens_zh))
        return alignment_matrix, words_en, tokens_zh_filtered

    en_mat = np.vstack(en_word_vecs).astype(np.float32)  # [n_en, hidden]
    zh_mat = zh_vecs.astype(np.float32)                  # [n_zh, hidden]

    # L2 归一化
    en_norms = np.linalg.norm(en_mat, axis=1, keepdims=True)
    zh_norms = np.linalg.norm(zh_mat, axis=1, keepdims=True)
    en_mat = en_mat / np.clip(en_norms, 1e-8, None)
    zh_mat = zh_mat / np.clip(zh_norms, 1e-8, None)

    # 点积 = 归一化后的余弦相似度
    alignment_matrix = en_mat @ zh_mat.T
    
    return alignment_matrix, words_en, tokens_zh_filtered


def compute_softmax_alignments(alignment_matrix):
    """
    对对齐矩阵分别按行和按列进行Softmax归一化
    
    Args:
        alignment_matrix: 词对齐相似度矩阵 (n_en_words, n_zh_tokens)
    
    Returns:
        softmax_en2zh: 按行Softmax - 英文词对中文token的概率分布 (每行和为1)
        softmax_zh2en: 按列Softmax - 中文token对英文词的概率分布 (每列和为1)
    """
    if alignment_matrix.size == 0:
        return alignment_matrix.copy(), alignment_matrix.copy()
    
    # 数值稳定的Softmax实现
    def stable_softmax(x, axis=None):
        """数值稳定的Softmax计算"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        return exp_x / sum_exp_x
    
    # 按行Softmax: 英文词→中文token的概率分布
    softmax_en2zh = stable_softmax(alignment_matrix, axis=1)
    
    # 按列Softmax: 中文token→英文词的概率分布  
    softmax_zh2en = stable_softmax(alignment_matrix, axis=0)
    
    return softmax_en2zh, softmax_zh2en


def compute_bidirectional_alignment(softmax_en2zh, softmax_zh2en, tokens_en, tokens_zh):
    """
    计算双向最大概率对齐的共同结果
    
    Args:
        softmax_en2zh: 英文词对中文token的概率分布矩阵
        softmax_zh2en: 中文token对英文词的概率分布矩阵
        tokens_en: 英文分词结果
        tokens_zh: 中文分词结果
    
    Returns:
        bidirectional_pairs: 双向一致对齐的词汇对列表 [(en_idx, zh_idx, prob_en2zh, prob_zh2en)]
    """
    bidirectional_pairs = []
    
    # 遍历所有英文词
    for i, token_en in enumerate(tokens_en):
        if i < softmax_en2zh.shape[0]:
            # 找出英文词i的最佳中文对齐
            best_zh_idx = np.argmax(softmax_en2zh[i, :])
            prob_en2zh = softmax_en2zh[i, best_zh_idx]
            
            # 检查该中文词的最佳英文对齐是否指向当前英文词
            if best_zh_idx < softmax_zh2en.shape[1]:
                best_en_idx = np.argmax(softmax_zh2en[:, best_zh_idx])
                prob_zh2en = softmax_zh2en[best_en_idx, best_zh_idx]
                
                # 如果双向互为最佳对齐，则记录
                if best_en_idx == i:
                    bidirectional_pairs.append((i, best_zh_idx, prob_en2zh, prob_zh2en))
    
    return bidirectional_pairs


def display_softmax_results(softmax_en2zh, softmax_zh2en, tokens_en, tokens_zh):
    """
    显示Softmax归一化后的概率分布矩阵和最佳对齐结果
    
    Args:
        softmax_en2zh: 英文词对中文token的概率分布矩阵
        softmax_zh2en: 中文token对英文词的概率分布矩阵  
        tokens_en: 英文分词结果
        tokens_zh: 中文分词结果
    """
    def format_token_for_display(token, max_len=12):
        display = str(token)
        display = display.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        if len(display) > max_len:
            display = display[:max_len-2] + '..'
        return display
    
    col_width = 12
    row_label_width = 24
    
    # 1. 显示英文→中文的概率分布矩阵 (按行Softmax)
    print(f"\n{'='*60}")
    print("英文词→中文token 概率分布矩阵 (按行Softmax，每行和=1):")
    print(f"{'='*60}")
    
    print(f"\n{'':{row_label_width}s}", end="")
    for j, token_zh in enumerate(tokens_zh):
        header = f"zh[{j}]-{format_token_for_display(token_zh, max_len=6)}"
        print(f"{header:>{col_width}s}", end="")
    print()
    
    print("-" * (row_label_width + col_width * len(tokens_zh)))
    
    for i, token_en in enumerate(tokens_en):
        row_label = f"en[{i}]-{format_token_for_display(token_en, 16)}"
        print(f"{row_label:<{row_label_width}s}", end="")
        
        for j in range(len(tokens_zh)):
            if i < softmax_en2zh.shape[0] and j < softmax_en2zh.shape[1]:
                prob = softmax_en2zh[i, j]
                if prob < 0.001:
                    print(f"{'0.000':>{col_width}s}", end="")
                else:
                    print(f"{prob:>{col_width}.3f}", end="")
            else:
                print(f"{'N/A':>{col_width}s}", end="")
        print()
    
    # 显示每个英文词的最佳对齐中文token
    print(f"\n英文词最大概率对齐:")
    print("-" * 50)
    for i, token_en in enumerate(tokens_en):
        if i < softmax_en2zh.shape[0]:
            best_idx = np.argmax(softmax_en2zh[i, :])
            best_prob = softmax_en2zh[i, best_idx]
            
            display_en = f"en[{i}]-{format_token_for_display(token_en, 20)}"
            display_zh = f"zh[{best_idx}]-{format_token_for_display(tokens_zh[best_idx], 20)}"
            print(f"  {display_en:28s} -> {display_zh:28s} (概率: {best_prob:.4f})")
    
    # 2. 显示中文→英文的概率分布矩阵 (按列Softmax)
    print(f"\n{'='*60}")
    print("中文token→英文词 概率分布矩阵 (按列Softmax，每列和=1):")
    print(f"{'='*60}")
    
    print(f"\n{'':{row_label_width}s}", end="")
    for j, token_zh in enumerate(tokens_zh):
        header = f"zh[{j}]-{format_token_for_display(token_zh, max_len=6)}"
        print(f"{header:>{col_width}s}", end="")
    print()
    
    print("-" * (row_label_width + col_width * len(tokens_zh)))
    
    for i, token_en in enumerate(tokens_en):
        row_label = f"en[{i}]-{format_token_for_display(token_en, 16)}"
        print(f"{row_label:<{row_label_width}s}", end="")
        
        for j in range(len(tokens_zh)):
            if i < softmax_zh2en.shape[0] and j < softmax_zh2en.shape[1]:
                prob = softmax_zh2en[i, j]
                if prob < 0.001:
                    print(f"{'0.000':>{col_width}s}", end="")
                else:
                    print(f"{prob:>{col_width}.3f}", end="")
            else:
                print(f"{'N/A':>{col_width}s}", end="")
        print()
    
    # 显示每个中文token的最佳对齐英文词
    print(f"\n中文token最大概率对齐:")
    print("-" * 50)
    for j, token_zh in enumerate(tokens_zh):
        if j < softmax_zh2en.shape[1]:
            best_idx = np.argmax(softmax_zh2en[:, j])
            best_prob = softmax_zh2en[best_idx, j]
            
            display_zh = f"zh[{j}]-{format_token_for_display(token_zh, 20)}"
            display_en = f"en[{best_idx}]-{format_token_for_display(tokens_en[best_idx], 20)}"
            print(f"  {display_zh:28s} -> {display_en:28s} (概率: {best_prob:.4f})")
    
    # 计算并显示双向最大概率对齐的共同结果
    print(f"\n{'='*60}")
    print("双向最大概率对齐的共同结果:")
    print(f"{'='*60}")
    
    bidirectional_pairs = compute_bidirectional_alignment(softmax_en2zh, softmax_zh2en, tokens_en, tokens_zh)
    
    if bidirectional_pairs:
        print(f"找到 {len(bidirectional_pairs)} 组双向一致的对齐词汇对:")
        print("-" * 70)
        for en_idx, zh_idx, prob_en2zh, prob_zh2en in bidirectional_pairs:
            display_en = f"en[{en_idx}]-{format_token_for_display(tokens_en[en_idx], 20)}"
            display_zh = f"zh[{zh_idx}]-{format_token_for_display(tokens_zh[zh_idx], 20)}"
            avg_prob = (prob_en2zh + prob_zh2en) / 2
            print(f"  {display_en:28s} <=> {display_zh:28s}")
            print(f"    英→中概率: {prob_en2zh:.4f} | 中→英概率: {prob_zh2en:.4f} | 平均概率: {avg_prob:.4f}")
            print()
    else:
        print("未找到双向一致的对齐词汇对。")
        print("这可能表明词汇对齐存在较大的不确定性或一对多/多对一的关系。")

    # 计算并输出英文句子的词覆盖率
    # 规则：分母=英文词汇总数（包含英文标点），分子=平均概率>0.075的双向一致对齐数
    threshold = 0.075
    total_en_words = len(tokens_en)
    matched_pairs = sum(1 for (en_idx, zh_idx, p_en, p_zh) in bidirectional_pairs
                        if ((p_en + p_zh) / 2) > threshold)
    coverage = (matched_pairs / total_en_words) if total_en_words > 0 else 0.0
    print("\n" + "-" * 60)
    print("英文句子的词覆盖率 (双向一致且平均概率>0.075 的对齐数 / 英文词汇总数):")
    print(f"  {matched_pairs}/{total_en_words} = {coverage:.2%}")


def display_alignment_results(alignment_matrix, tokens_en, tokens_zh):
    """
    完整显示词对齐相似度矩阵
    
    Args:
        alignment_matrix: 词对齐相似度矩阵
        tokens_en: 英文分词结果
        tokens_zh: 中文分词结果
    """
    print(f"\n{'='*60}")
    print("Token 分词相似度矩阵 (去除无关符号后的完整显示):")
    print(f"{'='*60}")
    
    # 输入此时已是（英文词列表, 已过滤的中文 token 列表），无需再展开
    tokens_en = tokens_en
    tokens_zh = tokens_zh
    
    # 准备处理显示 tokens（输入已为可显示文本，不再做二次解码）
    def format_token_for_display(token, max_len=12):
        display = str(token)
        display = display.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        if len(display) > max_len:
            display = display[:max_len-2] + '..'
        return display
    
    # 列/行显示宽度设置
    col_width = 12   # 每一列（中文 token 列）宽度
    row_label_width = 24  # 行标签（英文 token）的宽度
    
    # 显示矩阵头部 - 所有中文词（包含清理后索引）
    print(f"\n{'':{row_label_width}s}", end="")
    for j, token_zh in enumerate(tokens_zh):
        header = f"zh[{j}]-{format_token_for_display(token_zh, max_len=6)}"
        print(f"{header:>{col_width}s}", end="")
    print()
    
    print("-" * (row_label_width + col_width * len(tokens_zh)))
    
    # 显示矩阵内容 - 所有英文“词”（已聚合，包含清理后索引）
    for i, token_en in enumerate(tokens_en):
        row_label = f"en[{i}]-{format_token_for_display(token_en, 16)}"
        print(f"{row_label:<{row_label_width}s}", end="")
        
        for j in range(len(tokens_zh)):
            if i < alignment_matrix.shape[0] and j < alignment_matrix.shape[1]:
                score = alignment_matrix[i, j]
                # 使用更紧凑的格式
                if abs(score) < 0.0001:
                    print(f"{'0.0':>{col_width}s}", end="")
                else:
                    print(f"{score:>{col_width}.3f}", end="")
            else:
                print(f"{'N/A':>{col_width}s}", end="")
        print()
    
    # 找出每个英文词的最佳对齐（阈值 0.3）
    print(f"\n{'='*60}")
    print("最佳对齐结果 (相似度 > 0.3):")
    print(f"{'='*60}")
    for i, token_en in enumerate(tokens_en):
        # 英文侧此时均为词，已过滤特殊/标点，这里不再额外跳过
            
        if i < alignment_matrix.shape[0]:
            # 找出最佳对齐，但排除特殊标记
            valid_scores = []
            valid_indices = []
            for j in range(len(tokens_zh)):
                if True:
                    valid_scores.append(alignment_matrix[i, j])
                    valid_indices.append(j)
            
            if valid_scores:
                best_idx_in_valid = np.argmax(valid_scores)
                best_idx = valid_indices[best_idx_in_valid]
                best_score = alignment_matrix[i, best_idx]
                
                display_en = f"en[{i}]-{format_token_for_display(token_en, 20)}"
                display_zh = f"zh[{best_idx}]-{format_token_for_display(tokens_zh[best_idx], 20)}"
                
                # 只显示相似度较高的对齐（阈值 0.3）
                if best_score > 0.3:
                    print(f"  {display_en:28s} -> {display_zh:28s} (相似度: {best_score:.4f})")


# 主程序
if __name__ == "__main__":
    # 英文句子
    sentence_en = "Using premium fresh meat, welcome new and old teachers and students to dine."
    
    # 中文句子
    sentence_zh = "采用优等生鲜肉,欢迎新老师生前来就餐。"
    
    # 固定层号提示
    print(f"\n使用 {model_id} 模型第 {LAYER_NUM} 层进行词对齐分析")
    print("="*60)
    
    # 处理英文句子并显示分词结果
    embeddings_en, tokens_en = display_tokenization_and_vectors(
        model, tokenizer, sentence_en, "英文"
    )
    
    # 处理中文句子并显示分词结果
    embeddings_zh, tokens_zh = display_tokenization_and_vectors(
        model, tokenizer, sentence_zh, "中文"
    )
    
    # 计算词级别对齐（英文侧为“词”，中文侧保持子词/字级）
    alignment_matrix, words_en, tokens_zh_flat = compute_word_alignment(
        embeddings_en, embeddings_zh, tokens_en, tokens_zh
    )
    
    # 显示完整的对齐矩阵（行：英文词；列：中文 token）
    display_alignment_results(alignment_matrix, words_en, tokens_zh_flat)
    
    # 计算并显示Softmax归一化的概率分布
    print(f"\n{'='*60}")
    print("计算Softmax概率分布...")
    print(f"{'='*60}")
    
    softmax_en2zh, softmax_zh2en = compute_softmax_alignments(alignment_matrix)
    
    # 显示Softmax结果
    display_softmax_results(softmax_en2zh, softmax_zh2en, words_en, tokens_zh_flat)
    
    print("\n" + "="*60)
    print("分析完成!")


