#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地 facebook/xlm-roberta-xl 模型进行句子分割和对齐
功能：
- 对英文句子列表和中文句子列表进行双端收缩匹配
- 使用本地模型进行token分词和嵌入
- 计算双向一致的对齐token对和token覆盖度
- 过滤掉模型分词器的特殊标记符号
运行环境:
  pip install transformers torch numpy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import unicodedata

# 固定的编码层号常量
LAYER_NUM = 8

# 过滤阈值常量定义
FILTER_THRESHOLD_EN = 0.16  # 英文源句子作为分母时的过滤阈值
FILTER_THRESHOLD_ZH = 0.18  # 中文源句子作为分母时的过滤阈值

# 标签常量定义 - 更直白的命名
LABEL_TARGET_SEGMENT = "Target_segment"
LABEL_SOURCE_EN = "Source_en"
LABEL_SOURCE_ZH = "Source_zh"

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
print("[处理说明] 对目标句子进行前向和后向的分割，计算分割出的候选子句相对于源句列表的词覆盖度")
print("="*60)


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


def filter_valid_tokens(tokens: list, embedding_tensor: torch.Tensor):
    """
    过滤掉特殊标记，保留有效的token及其嵌入向量
    
    Args:
        tokens: 原始token列表
        embedding_tensor: 对应的嵌入向量张量
    
    Returns:
        token_texts: 过滤后的token文本列表
        kept_indices: 保留的token索引列表  
        kept_embeddings: 过滤后的嵌入向量张量
    """
    kept_indices = []
    token_texts = []
    
    for i, token in enumerate(tokens):
        if not is_special_token(token):
            kept_indices.append(i)
            token_texts.append(decode_token(token))
    
    if len(kept_indices) == 0:
        kept_embeddings = torch.empty((0, embedding_tensor.shape[-1]), device=embedding_tensor.device)
    else:
        kept_embeddings = embedding_tensor[kept_indices]
    
    return token_texts, kept_indices, kept_embeddings





def compute_token_alignment(embeddings_en, embeddings_zh, tokens_en, tokens_zh):
    """
    计算中英文句子的token级别对齐和相似度。

    Args:
        embeddings_en: 英文句子的token级嵌入 (seq_len, hidden)
        embeddings_zh: 中文句子的token级嵌入 (seq_len, hidden)
        tokens_en: 英文分词结果（token列表，或 [token列表] 的嵌套）
        tokens_zh: 中文分词结果（token列表，或 [token列表] 的嵌套）

    Returns:
        alignment_matrix: token对齐相似度矩阵（英文token × 中文token）
        tokens_en_filtered: 英文token列表（与矩阵行对应）
        tokens_zh_filtered: 中文token列表（与矩阵列对应）
    """
    # 获取token级向量
    vec_en = embeddings_en[0] if isinstance(embeddings_en, list) else embeddings_en
    vec_zh = embeddings_zh[0] if isinstance(embeddings_zh, list) else embeddings_zh

    # 转换为 numpy 数组
    if isinstance(vec_en, torch.Tensor):
        vec_en = vec_en.cpu().numpy()
    if isinstance(vec_zh, torch.Tensor):
        vec_zh = vec_zh.cpu().numpy()

    # 展平 tokens 列表
    tokens_en_list = tokens_en[0] if isinstance(tokens_en[0], list) else tokens_en
    tokens_zh_list = tokens_zh[0] if isinstance(tokens_zh[0], list) else tokens_zh

    # 英文：过滤特殊标记，保留有效token
    tokens_en_filtered, en_kept_indices, _ = filter_valid_tokens(tokens_en_list, vec_en)
    en_vecs = vec_en[en_kept_indices] if len(en_kept_indices) > 0 else np.zeros((0, vec_en.shape[1]))

    # 中文：过滤特殊标记，保留有效token  
    tokens_zh_filtered, zh_kept_indices, _ = filter_valid_tokens(tokens_zh_list, vec_zh)
    zh_vecs = vec_zh[zh_kept_indices] if len(zh_kept_indices) > 0 else np.zeros((0, vec_zh.shape[1]))

    # 计算"英文token × 中文token"的相似度矩阵（先进行 L2 归一化，再用点积，等价于余弦相似度）
    n_tokens_en = len(tokens_en_filtered)
    n_tokens_zh = len(tokens_zh_filtered)
    
    if n_tokens_en == 0 or n_tokens_zh == 0:
        alignment_matrix = np.zeros((n_tokens_en, n_tokens_zh))
        return alignment_matrix, tokens_en_filtered, tokens_zh_filtered

    en_mat = en_vecs.astype(np.float32)  # [n_en, hidden]
    zh_mat = zh_vecs.astype(np.float32)  # [n_zh, hidden]

    # L2 归一化
    en_norms = np.linalg.norm(en_mat, axis=1, keepdims=True)
    zh_norms = np.linalg.norm(zh_mat, axis=1, keepdims=True)
    en_mat = en_mat / np.clip(en_norms, 1e-8, None)
    zh_mat = zh_mat / np.clip(zh_norms, 1e-8, None)

    # 点积 = 归一化后的余弦相似度
    alignment_matrix = en_mat @ zh_mat.T

    return alignment_matrix, tokens_en_filtered, tokens_zh_filtered


def compute_softmax_alignments(alignment_matrix):
    """
    对对齐矩阵分别按行和按列进行Softmax归一化

    Args:
        alignment_matrix: token对齐相似度矩阵 (n_en_tokens, n_zh_tokens)

    Returns:
        softmax_en2zh: 按行Softmax - 英文token对中文token的概率分布 (每行和为1)
        softmax_zh2en: 按列Softmax - 中文token对英文token的概率分布 (每列和为1)
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

    # 按行Softmax: 英文token→中文token的概率分布
    softmax_en2zh = stable_softmax(alignment_matrix, axis=1)

    # 按列Softmax: 中文token→英文token的概率分布
    softmax_zh2en = stable_softmax(alignment_matrix, axis=0)

    return softmax_en2zh, softmax_zh2en


def compute_bidirectional_alignment(softmax_en2zh, softmax_zh2en, tokens_en, tokens_zh):
    """
    计算双向最大概率对齐的共同结果

    Args:
        softmax_en2zh: 英文token对中文token的概率分布矩阵
        softmax_zh2en: 中文token对英文token的概率分布矩阵
        tokens_en: 英文分词结果
        tokens_zh: 中文分词结果

    Returns:
        bidirectional_pairs: 双向一致对齐的token对列表 [(en_idx, zh_idx, prob_en2zh, prob_zh2en)]
    """
    bidirectional_pairs = []

    # 遍历所有英文token
    for i, token_en in enumerate(tokens_en):
        if i < softmax_en2zh.shape[0]:
            # 找出英文tokeni的最佳中文对齐
            best_zh_idx = np.argmax(softmax_en2zh[i, :])
            prob_en2zh = softmax_en2zh[i, best_zh_idx]

            # 检查该中文token的最佳英文对齐是否指向当前英文token
            if best_zh_idx < softmax_zh2en.shape[1]:
                best_en_idx = np.argmax(softmax_zh2en[:, best_zh_idx])
                prob_zh2en = softmax_zh2en[best_en_idx, best_zh_idx]

                # 如果双向互为最佳对齐，则记录
                if best_en_idx == i:
                    bidirectional_pairs.append((i, best_zh_idx, prob_en2zh, prob_zh2en))

    return bidirectional_pairs


def _compute_token_coverage_core_with_context(softmax_1to2, softmax_2to1, tokens_1_filtered, tokens_2_filtered,
                                            embeddings_1, embeddings_2, kept_indices_1, kept_indices_2,
                                            context_label1, context_label2):
    """
    带上下文标签的核心token覆盖度计算逻辑

    Args:
        softmax_1to2: 第一个句子token对第二个句子token的概率分布矩阵
        softmax_2to1: 第二个句子token对第一个句子token的概率分布矩阵
        tokens_1_filtered: 第一个句子的有效token文本列表
        tokens_2_filtered: 第二个句子的有效token文本列表
        embeddings_1: 第一个句子的嵌入向量（numpy数组）
        embeddings_2: 第二个句子的嵌入向量（numpy数组）
        kept_indices_1: 第一个句子有效token的索引列表
        kept_indices_2: 第二个句子有效token的索引列表
        context_label1: 第一个句子的上下文标签
        context_label2: 第二个句子的上下文标签

    Returns:
        coverage: token覆盖度 (0.0 到 1.0)
        bidirectional_pairs: 双向一致对齐的token对
    """
    # 计算双向一致对齐
    bidirectional_pairs = compute_bidirectional_alignment(softmax_1to2, softmax_2to1, tokens_1_filtered, tokens_2_filtered)

    # 计算token数
    total_tokens_1 = len(tokens_1_filtered)
    total_tokens_2 = len(tokens_2_filtered)

    # 显示参与计算的值
    print(f"    [覆盖度计算] {context_label1} token数: {total_tokens_1}, {context_label2} token数: {total_tokens_2}")

    # 根据context_label1决定使用哪个方向的概率和过滤阈值
    if context_label1 == LABEL_SOURCE_EN:
        # 英文源句子与候选子句对比：使用英文源句子的行softmax概率
        filter_threshold = FILTER_THRESHOLD_EN
        print(f"    按英文列表句的行softmax概率 过滤阈值: {filter_threshold}")
        use_p_1to2 = True  # 使用第一个句子（英文源句子）对第二个句子（候选子句）的概率
    elif context_label1 == LABEL_SOURCE_ZH:
        # 中文源句子与候选子句对比：使用中文源句子的行softmax概率
        filter_threshold = FILTER_THRESHOLD_ZH
        print(f"    按中文列表句的行softmax概率 过滤阈值: {filter_threshold}")
        use_p_1to2 = True  # 使用第一个句子（中文源句子）对第二个句子（候选子句）的概率
    else:
        # 其他情况保持原有逻辑，默认使用英文阈值
        filter_threshold = FILTER_THRESHOLD_EN
        print(f"    {context_label1}列表  按{context_label1}列表的行softmax概率 过滤阈值: {filter_threshold}")
        use_p_1to2 = True

    # 分析双语一致性token对并收集有效余弦相似度
    cosine_similarities = []

    print(f"    [一致性分析] 双向一致对齐 token 对:")
    for idx_1, idx_2, p_1to2, p_2to1 in bidirectional_pairs:
        token_1 = tokens_1_filtered[idx_1] if idx_1 < len(tokens_1_filtered) else "N/A"
        token_2 = tokens_2_filtered[idx_2] if idx_2 < len(tokens_2_filtered) else "N/A"

        # 根据设置选择使用哪个方向的概率进行过滤
        filter_prob = p_1to2 if use_p_1to2 else p_2to1
        if filter_prob > filter_threshold:
            # 获取对应的原始嵌入向量
            if idx_1 < len(kept_indices_1) and idx_2 < len(kept_indices_2):
                embedding_1 = embeddings_1[kept_indices_1[idx_1]]  # 第一个句子token的嵌入向量
                embedding_2 = embeddings_2[kept_indices_2[idx_2]]  # 第二个句子token的嵌入向量

                # L2归一化
                norm_1 = np.linalg.norm(embedding_1)
                norm_2 = np.linalg.norm(embedding_2)

                if norm_1 > 1e-8 and norm_2 > 1e-8:
                    normalized_1 = embedding_1 / norm_1
                    normalized_2 = embedding_2 / norm_2

                    # 计算余弦相似度（L2归一化后的点积）
                    cosine_sim = np.dot(normalized_1, normalized_2)
                    cosine_similarities.append(cosine_sim)

                    # 根据使用的概率方向生成正确的输出描述
                    if context_label1 == LABEL_SOURCE_EN:
                        print(f"      ✓ '{token_1}' <-> '{token_2}' (按英文列表句的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 余弦相似度: {cosine_sim:.3f})")
                    elif context_label1 == LABEL_SOURCE_ZH:
                        print(f"      ✓ '{token_1}' <-> '{token_2}' (按中文列表句的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 余弦相似度: {cosine_sim:.3f})")
                    else:
                        print(f"      ✓ '{token_1}' <-> '{token_2}' (按{context_label1}列表的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 余弦相似度: {cosine_sim:.3f})")
                else:
                    print(f"      ✗ '{token_1}' <-> '{token_2}' (向量归一化失败)")
            else:
                print(f"      ✗ '{token_1}' <-> '{token_2}' (索引越界)")
        else:
            # 根据使用的概率方向生成正确的输出描述
            if context_label1 == LABEL_SOURCE_EN:
                print(f"      ✗ '{token_1}' <-> '{token_2}' (按英文列表句的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 未通过阈值过滤)")
            elif context_label1 == LABEL_SOURCE_ZH:
                print(f"      ✗ '{token_1}' <-> '{token_2}' (按中文列表句的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 未通过阈值过滤)")
            else:
                print(f"      ✗ '{token_1}' <-> '{token_2}' (按{context_label1}列表的行softmax概率: {p_1to2:.3f}, 按{context_label2}的列softmax概率: {p_2to1:.3f}, 未通过阈值过滤)")

    print(f"    [过滤结果] 符合阈值的对齐数: {len(cosine_similarities)}/{len(bidirectional_pairs)}")

    # 过滤后的余弦相似度累加计算token覆盖度：累加值/第一个句子token数
    cosine_sum = sum(cosine_similarities) if cosine_similarities else 0.0
    if total_tokens_1 > 0 and cosine_similarities:
        coverage = cosine_sum / total_tokens_1
    else:
        coverage = 0.0

    print(f"    [通过过滤的余弦相似度累加]: {cosine_sum:.3f}")
    print(f"    [覆盖度结果] {cosine_sum:.3f} / {total_tokens_1} = {coverage:.3f}")

    return coverage, bidirectional_pairs


def _compute_token_coverage_core(softmax_en2zh, softmax_zh2en, tokens_en_filtered, tokens_zh_filtered,
                                en_embeddings, zh_embeddings, en_kept_indices, zh_kept_indices):
    """
    核心的token覆盖度计算逻辑（避免重复代码）

    Args:
        softmax_en2zh: 英文token对中文token的概率分布矩阵
        softmax_zh2en: 中文token对英文token的概率分布矩阵
        tokens_en_filtered: 英文的有效token文本列表
        tokens_zh_filtered: 中文的有效token文本列表
        en_embeddings: 英文的嵌入向量（numpy数组）
        zh_embeddings: 中文的嵌入向量（numpy数组）
        en_kept_indices: 英文有效token的索引列表
        zh_kept_indices: 中文有效token的索引列表

    Returns:
        coverage: token覆盖度 (0.0 到 1.0)
        bidirectional_pairs: 双向一致对齐的token对
    """
    # 计算双向一致对齐
    bidirectional_pairs = compute_bidirectional_alignment(softmax_en2zh, softmax_zh2en, tokens_en_filtered, tokens_zh_filtered)

    # 计算英文token数和中文token数
    total_en_tokens = len(tokens_en_filtered)
    total_zh_tokens = len(tokens_zh_filtered)

    # 显示参与计算的值
    print(f"    [覆盖度计算] 英文 token数: {total_en_tokens}, 中文 token数: {total_zh_tokens}")

    # 过滤方法：过滤阈值英→中概率覆盖度计算（英文作为源句子）
    filter_threshold = FILTER_THRESHOLD_EN
    print(f"   过滤阈值: {filter_threshold}")

    # 分析双语一致性token对并收集有效余弦相似度
    cosine_similarities = []

    print(f"    [一致性分析] 双向一致对齐 token 对:")
    for en_idx, zh_idx, p_en, p_zh in bidirectional_pairs:
        en_token = tokens_en_filtered[en_idx] if en_idx < len(tokens_en_filtered) else "N/A"
        zh_token = tokens_zh_filtered[zh_idx] if zh_idx < len(tokens_zh_filtered) else "N/A"

        # 过滤：只保留英→中概率大于阈值的对齐
        if p_en > filter_threshold:
            # 获取对应的原始嵌入向量
            if en_idx < len(en_kept_indices) and zh_idx < len(zh_kept_indices):
                en_embedding = en_embeddings[en_kept_indices[en_idx]]  # 英文token的嵌入向量
                zh_embedding = zh_embeddings[zh_kept_indices[zh_idx]]  # 中文token的嵌入向量
                
                # L2归一化
                en_norm = np.linalg.norm(en_embedding)
                zh_norm = np.linalg.norm(zh_embedding)
                
                if en_norm > 1e-8 and zh_norm > 1e-8:
                    en_normalized = en_embedding / en_norm
                    zh_normalized = zh_embedding / zh_norm
                    
                    # 计算余弦相似度（L2归一化后的点积）
                    cosine_sim = np.dot(en_normalized, zh_normalized)
                    cosine_similarities.append(cosine_sim)
                    
                    print(f"      ✓ '{en_token}' <-> '{zh_token}' (英→中: {p_en:.3f}, 中→英: {p_zh:.3f}, 余弦相似度: {cosine_sim:.3f})")
                else:
                    print(f"      ✗ '{en_token}' <-> '{zh_token}' (向量归一化失败)")
            else:
                print(f"      ✗ '{en_token}' <-> '{zh_token}' (索引越界)")
        else:
            print(f"      ✗ '{en_token}' <-> '{zh_token}' (英→中: {p_en:.3f}, 中→英: {p_zh:.3f}, 未通过英→中阈值过滤)")

    print(f"    [过滤结果] 符合阈值的对齐数: {len(cosine_similarities)}/{len(bidirectional_pairs)}")

    # 过滤后的余弦相似度累加计算token覆盖度：累加值/英文token数  
    cosine_sum = sum(cosine_similarities) if cosine_similarities else 0.0
    if total_en_tokens > 0 and cosine_similarities:
        coverage = cosine_sum / total_en_tokens
    else:
        coverage = 0.0

    print(f"    [通过过滤的余弦相似度累加]: {cosine_sum:.3f}")
    print(f"    [覆盖度结果] {cosine_sum:.3f} / {total_en_tokens} = {coverage:.3f}")

    return coverage, bidirectional_pairs


def calculate_token_coverage_with_context(sentence1, sentence2, context_label1, context_label2):
    """
    计算token覆盖度并提供上下文标签的输出

    Args:
        sentence1: 第一个句子字符串
        sentence2: 第二个句子字符串
        context_label1: 第一个句子的上下文标签
        context_label2: 第二个句子的上下文标签

    Returns:
        coverage: token覆盖度 (0.0 到 1.0)
        bidirectional_pairs: 双向一致对齐的token对
    """
    # 获取token嵌入和分词结果
    embeddings_1, tokens_1 = get_layer_embeddings(model, tokenizer, sentence1)
    embeddings_2, tokens_2 = get_layer_embeddings(model, tokenizer, sentence2)

    # 计算token级别对齐
    alignment_matrix, tokens_1_filtered, tokens_2_filtered = compute_token_alignment(
        embeddings_1, embeddings_2, tokens_1, tokens_2
    )

    # 计算Softmax概率分布
    softmax_1to2, softmax_2to1 = compute_softmax_alignments(alignment_matrix)

    # 获取过滤后的原始嵌入向量用于余弦相似度计算
    vec_1 = embeddings_1[0] if isinstance(embeddings_1, list) else embeddings_1
    vec_2 = embeddings_2[0] if isinstance(embeddings_2, list) else embeddings_2

    # 转换为 numpy 数组
    if isinstance(vec_1, torch.Tensor):
        vec_1 = vec_1.cpu().numpy()
    if isinstance(vec_2, torch.Tensor):
        vec_2 = vec_2.cpu().numpy()

    # 展平 tokens 列表
    tokens_1_list = tokens_1[0] if isinstance(tokens_1[0], list) else tokens_1
    tokens_2_list = tokens_2[0] if isinstance(tokens_2[0], list) else tokens_2

    # 获取过滤后的嵌入向量索引
    _, kept_indices_1, _ = filter_valid_tokens(tokens_1_list, vec_1)
    _, kept_indices_2, _ = filter_valid_tokens(tokens_2_list, vec_2)

    # 调用核心计算函数，传递上下文标签
    return _compute_token_coverage_core_with_context(
        softmax_1to2, softmax_2to1, tokens_1_filtered, tokens_2_filtered,
        vec_1, vec_2, kept_indices_1, kept_indices_2, context_label1, context_label2
    )


def calculate_token_coverage(english_sentence, chinese_sentence):
    """
    计算英文句子的token覆盖度

    Args:
        english_sentence: 英文句子字符串
        chinese_sentence: 中文句子字符串

    Returns:
        coverage: token覆盖度 (0.0 到 1.0)
        bidirectional_pairs: 双向一致对齐的token对
    """
    # 获取token嵌入和分词结果
    embeddings_en, tokens_en = get_layer_embeddings(model, tokenizer, english_sentence)
    embeddings_zh, tokens_zh = get_layer_embeddings(model, tokenizer, chinese_sentence)

    # 计算token级别对齐
    alignment_matrix, tokens_en_filtered, tokens_zh_filtered = compute_token_alignment(
        embeddings_en, embeddings_zh, tokens_en, tokens_zh
    )

    # 计算Softmax概率分布
    softmax_en2zh, softmax_zh2en = compute_softmax_alignments(alignment_matrix)

    # 获取过滤后的原始嵌入向量用于余弦相似度计算
    vec_en = embeddings_en[0] if isinstance(embeddings_en, list) else embeddings_en
    vec_zh = embeddings_zh[0] if isinstance(embeddings_zh, list) else embeddings_zh
    
    # 转换为 numpy 数组
    if isinstance(vec_en, torch.Tensor):
        vec_en = vec_en.cpu().numpy()
    if isinstance(vec_zh, torch.Tensor):
        vec_zh = vec_zh.cpu().numpy()

    # 展平 tokens 列表
    tokens_en_list = tokens_en[0] if isinstance(tokens_en[0], list) else tokens_en
    tokens_zh_list = tokens_zh[0] if isinstance(tokens_zh[0], list) else tokens_zh

    # 获取过滤后的嵌入向量索引
    _, en_kept_indices, _ = filter_valid_tokens(tokens_en_list, vec_en)
    _, zh_kept_indices, _ = filter_valid_tokens(tokens_zh_list, vec_zh)

    # 调用核心计算函数
    return _compute_token_coverage_core(softmax_en2zh, softmax_zh2en, tokens_en_filtered, tokens_zh_filtered,
                                       vec_en, vec_zh, en_kept_indices, zh_kept_indices)


def calculate_token_coverage_with_precomputed(english_sentence, zh_embeddings, zh_tokens_filtered, zh_filtered_indices_subset):
    """
    使用预计算的中文嵌入计算英文句子的token覆盖度
    
    Args:
        english_sentence: 英文句子字符串
        zh_embeddings: 中文句子的预计算嵌入向量（已过滤的子集）
        zh_tokens_filtered: 中文的有效token文本列表（已过滤的子集）
        zh_filtered_indices_subset: 当前使用的中文token在全句中的索引
    
    Returns:
        coverage: token覆盖度 (0.0 到 1.0)
        bidirectional_pairs: 双向一致对齐的token对
    """
    # 获取英文句子的token嵌入和分词结果
    embeddings_en, tokens_en = get_layer_embeddings(model, tokenizer, english_sentence)
    
    # 过滤英文的特殊标记
    tokens_en_list = tokens_en[0] if isinstance(tokens_en[0], list) else tokens_en
    tokens_en_filtered, en_kept_indices, _ = filter_valid_tokens(tokens_en_list, embeddings_en[0])
    
    # 获取英文的有效嵌入向量
    en_embeddings = embeddings_en[0][en_kept_indices] if len(en_kept_indices) > 0 else torch.empty((0, embeddings_en[0].shape[-1]), device=embeddings_en[0].device)
    
    # 计算对齐矩阵
    if len(tokens_en_filtered) == 0 or len(zh_tokens_filtered) == 0:
        alignment_matrix = np.zeros((len(tokens_en_filtered), len(zh_tokens_filtered)))
    else:
        # 转换为numpy并进行L2归一化
        en_mat = en_embeddings.cpu().numpy().astype(np.float32)
        zh_mat = zh_embeddings.cpu().numpy().astype(np.float32)
        
        # L2 归一化
        en_norms = np.linalg.norm(en_mat, axis=1, keepdims=True)
        zh_norms = np.linalg.norm(zh_mat, axis=1, keepdims=True)
        en_mat = en_mat / np.clip(en_norms, 1e-8, None)
        zh_mat = zh_mat / np.clip(zh_norms, 1e-8, None)
        
        # 点积 = 归一化后的余弦相似度
        alignment_matrix = en_mat @ zh_mat.T

    # 计算Softmax概率分布
    softmax_en2zh, softmax_zh2en = compute_softmax_alignments(alignment_matrix)

    # 转换嵌入向量为numpy数组以适配核心函数
    en_embeddings_np = en_embeddings.cpu().numpy() if isinstance(en_embeddings, torch.Tensor) else en_embeddings
    zh_embeddings_np = zh_embeddings.cpu().numpy() if isinstance(zh_embeddings, torch.Tensor) else zh_embeddings
    
    # 为核心函数创建假的索引（因为嵌入向量已经是过滤后的）
    en_kept_indices = list(range(len(tokens_en_filtered)))
    zh_kept_indices = list(range(len(zh_tokens_filtered)))

    # 调用核心计算函数
    return _compute_token_coverage_core(softmax_en2zh, softmax_zh2en, tokens_en_filtered, zh_tokens_filtered,
                                       en_embeddings_np, zh_embeddings_np, en_kept_indices, zh_kept_indices)







def find_optimal_segmentation(Source_sentences_en, Target_sentence_segmentation, Source_sentences_zh):
    """
    对 Target_sentence_segmentation 目标句子分割 寻找 前向/后向 候选子句的的最佳分割点，使得整体token覆盖度最大

    Args:
        Source_sentences_en: Source_sentences_en目标英文列表 (2个或3个句子)
        Target_sentence_segmentation: Target_sentence_segmentation目标句子分割字符串
        Source_sentences_zh: Source_sentences_zh目标中文列表，用于对比评判

    Returns:
        best_segmentation: 最佳分割结果
    """
    # 对Target_sentence_segmentation目标句子分割进行分词以获取token边界，但每个候选子句将独立编码
    embeddings_zh, tokens_zh = get_layer_embeddings(model, tokenizer, Target_sentence_segmentation)
    chinese_tokens = tokens_zh[0]  # 获取分词结果

    # 过滤特殊标记，记录有效token的文本
    chinese_filtered_tokens = []

    for i, token in enumerate(chinese_tokens):
        if not is_special_token(token):
            chinese_filtered_tokens.append(decode_token(token))



    num_english = len(Source_sentences_en)
    num_chinese_tokens = len(chinese_filtered_tokens)

    if num_english == 2:
        return find_best_two_way_split(Source_sentences_en, chinese_filtered_tokens, Target_sentence_segmentation, Source_sentences_zh)
    else:
        print(f"错误：只支持2个Source_sentences_en目标英文列表句子，当前有{num_english}个句子")
        return None


def find_best_two_way_split(Source_sentences_en, chinese_tokens, Target_sentence_segmentation, Source_sentences_zh):
    """
    对 Target_sentence_segmentation 目标句子分割 寻找 前向/后向 候选子句的的最佳分割点
    对每个候选子句进行独立编码
    同时计算候选子句与Source_sentences_zh目标中文列表的覆盖度

    Args:
        Source_sentences_en: Source_sentences_en目标英文列表
        chinese_tokens: 中文分词结果（用于确定候选子句边界）
        Target_sentence_segmentation: Target_sentence_segmentation目标句子分割原始字符串
        Source_sentences_zh: Source_sentences_zh目标中文列表
    """
    print(f"\n{'='*60}")
    print("[前向分割] 开始处理前向候选子句 - 从目标句子开头逐步增加token长度")
    print(f"{'='*60}")

    # 为Source_sentences_en目标英文列表第一句寻找最佳覆盖度
    Source_sentences_en_first = Source_sentences_en[0]
    
    # 分别追踪英文和中文的最佳覆盖度位置
    best_coverage_first_en = 0.0
    best_split_for_first_en = None
    best_segment_first_en = None
    best_pairs_first_en = None

    best_coverage_first_zh = 0.0
    best_split_for_first_zh = None
    best_segment_first_zh = None
    best_pairs_first_zh = None

    for split_pos in range(1, len(chinese_tokens)):
        # 生成候选子句字符串
        chinese_tokens_subset = chinese_tokens[:split_pos]
        Target_sentence_segmentation_1 = ''.join(chinese_tokens_subset)

        print(f"\n前向-候选子句{split_pos}: '{Target_sentence_segmentation_1}'    {'='*20}")

        # 第一部分：与Source_sentences_en目标英文列表第一句的覆盖度计算
        print(f"   与Source_sentences_en目标英文列表第一句的覆盖度计算:")
        coverage_first, pairs_first = calculate_token_coverage_with_context(
            Source_sentences_en_first,
            Target_sentence_segmentation_1,
            LABEL_SOURCE_EN,
            LABEL_TARGET_SEGMENT
        )
        print(f"    与Source_sentences_en目标英文列表第一句的覆盖度: {coverage_first:.3f}")

        # 分隔线
        print(f"   {'='*21}")

        # 第二部分：与Source_sentences_zh目标中文列表第一句的覆盖度计算
        print(f"   与Source_sentences_zh目标中文列表第一句的覆盖度计算:")
        Source_sentences_zh_first = Source_sentences_zh[0]
        coverage_first_vs_chinese_ref, pairs_first_vs_chinese_ref = calculate_token_coverage_with_context(
            Source_sentences_zh_first,
            Target_sentence_segmentation_1,
            LABEL_SOURCE_ZH,
            LABEL_TARGET_SEGMENT
        )
        print(f"    与Source_sentences_zh目标中文列表第一句的覆盖度: {coverage_first_vs_chinese_ref:.3f}")

        # 分别更新英文和中文的最佳覆盖度
        if coverage_first > best_coverage_first_en:
            best_coverage_first_en = coverage_first
            best_split_for_first_en = split_pos
            best_segment_first_en = Target_sentence_segmentation_1
            best_pairs_first_en = pairs_first

        if coverage_first_vs_chinese_ref > best_coverage_first_zh:
            best_coverage_first_zh = coverage_first_vs_chinese_ref
            best_split_for_first_zh = split_pos
            best_segment_first_zh = Target_sentence_segmentation_1
            best_pairs_first_zh = pairs_first_vs_chinese_ref


    


    print(f"\n{'='*60}")
    print("[后向分割] 开始处理后向候选子句 - 从目标句子末尾逐步增加token长度")
    print(f"{'='*60}")

    # 为Source_sentences_en目标英文列表最后一句寻找最佳覆盖度（从尾部向前增加）
    Source_sentences_en_last = Source_sentences_en[1]
    
    # 分别追踪英文和中文的最佳覆盖度位置
    best_coverage_last_en = 0.0
    best_length_for_last_en = None
    best_segment_last_en = None
    best_pairs_last_en = None

    best_coverage_last_zh = 0.0
    best_length_for_last_zh = None
    best_segment_last_zh = None
    best_pairs_last_zh = None

    # 从尾部开始，逐步向前增加分词长度
    for length in range(1, len(chinese_tokens) + 1):
        # 生成候选子句字符串
        chinese_tokens_subset = chinese_tokens[-length:]
        Target_sentence_segmentation_2 = ''.join(chinese_tokens_subset)

        print(f"\n后向-候选子句{length}: '{Target_sentence_segmentation_2}'    {'='*20}")

        # 第一部分：与Source_sentences_en目标英文列表最后一句的覆盖度计算
        print(f"   与Source_sentences_en目标英文列表最后一句的覆盖度计算:")
        coverage2, pairs2 = calculate_token_coverage_with_context(
            Source_sentences_en_last,
            Target_sentence_segmentation_2,
            LABEL_SOURCE_EN,
            LABEL_TARGET_SEGMENT
        )
        print(f"    与Source_sentences_en目标英文列表最后一句的覆盖度: {coverage2:.3f}")

        # 分隔线
        print(f"   {'='*21}")

        # 第二部分：与Source_sentences_zh目标中文列表最后一句的覆盖度计算
        print(f"   与Source_sentences_zh目标中文列表最后一句的覆盖度计算:")
        Source_sentences_zh_last = Source_sentences_zh[-1]
        coverage2_vs_chinese_ref, pairs2_vs_chinese_ref = calculate_token_coverage_with_context(
            Source_sentences_zh_last,
            Target_sentence_segmentation_2,
            LABEL_SOURCE_ZH,
            LABEL_TARGET_SEGMENT
        )
        print(f"    与Source_sentences_zh目标中文列表最后一句的覆盖度: {coverage2_vs_chinese_ref:.3f}")

        # 分别更新英文和中文的最佳覆盖度
        if coverage2 > best_coverage_last_en:
            best_coverage_last_en = coverage2
            best_length_for_last_en = length
            best_segment_last_en = Target_sentence_segmentation_2
            best_pairs_last_en = pairs2

        if coverage2_vs_chinese_ref > best_coverage_last_zh:
            best_coverage_last_zh = coverage2_vs_chinese_ref
            best_length_for_last_zh = length
            best_segment_last_zh = Target_sentence_segmentation_2
            best_pairs_last_zh = pairs2_vs_chinese_ref


    


    result = {
        'first_sentence': {
            'en_best': {
                'split_position': best_split_for_first_en,
                'segment': best_segment_first_en,
                'coverage': best_coverage_first_en,
                'pairs': best_pairs_first_en
            },
            'zh_best': {
                'split_position': best_split_for_first_zh,
                'segment': best_segment_first_zh,
                'coverage': best_coverage_first_zh,
                'pairs': best_pairs_first_zh
            }
        },
        'last_sentence': {
            'en_best': {
                'tail_length': best_length_for_last_en,
                'segment': best_segment_last_en,
                'coverage': best_coverage_last_en,
                'pairs': best_pairs_last_en
            },
            'zh_best': {
                'tail_length': best_length_for_last_zh,
                'segment': best_segment_last_zh,
                'coverage': best_coverage_last_zh,
                'pairs': best_pairs_last_zh
            }
        }
    }

    return result



def main():
    # 原始测试用例：2个句子
    #Source_sentences_en目标英文列表
    Source_sentences_en = [
        # "in which the everyday citizens in these communities",
        # "contribute to the projects that are in the campaign"

        "is that it leverages squares and square roots",
        "in order to determine the matching amounts."
    ]
    #Source_sentences_zh目标中文列表
    Source_sentences_zh = [
        # "这些社区的普通居民",
        # "为活动中的项目捐款"

        "它利用了平方和平方根",
        "为了确定匹配金额。"
    ]
    # Target_sentence_segmentation 目标句子分割 要分割成前向/后向不同候选子句,并与Source_sentences_en目标英文列表和Source_sentences_zh目标中文列表 进行 前向-首句 后向-尾句 的对比 在寻找最佳的分割/覆盖度位置
    Target_sentence_segmentation =  "是因为它利用平方和平方根来确定匹配金额。"  #"由社区里的普通市民为这些活动中的项目出资"


    # 寻找最佳分割位置
    result = find_optimal_segmentation(Source_sentences_en, Target_sentence_segmentation, Source_sentences_zh)

    if result:
        print(f"\n{'='*60}")
        print("前向和后向候选子句的最佳 token 覆盖度结果")
        print(f"{'='*60}")

        # 前向候选子句结果
        print(f"\n[前向候选子句] 分别计算最佳覆盖度位置:")
        Source_sentences_en_first = Source_sentences_en[0]
        Source_sentences_zh_first = Source_sentences_zh[0]

        print(f"  目标英文最佳覆盖度位置:")
        print(f"    候选子句位置: {result['first_sentence']['en_best']['split_position']}")
        print(f"    最佳匹配片段: '{result['first_sentence']['en_best']['segment']}'")
        print(f"    覆盖度: {result['first_sentence']['en_best']['coverage']:.3f}")
        print(f"    对应英文句子: '{Source_sentences_en_first}'")

        print(f"  目标中文最佳覆盖度位置:")
        print(f"    候选子句位置: {result['first_sentence']['zh_best']['split_position']}")
        print(f"    最佳匹配片段: '{result['first_sentence']['zh_best']['segment']}'")
        print(f"    覆盖度: {result['first_sentence']['zh_best']['coverage']:.3f}")
        print(f"    对应中文句子: '{Source_sentences_zh_first}'")

        # 后向候选子句结果
        print(f"\n[后向候选子句] 分别计算最佳覆盖度位置:")
        Source_sentences_en_last = Source_sentences_en[1]
        Source_sentences_zh_last = Source_sentences_zh[-1]

        print(f"  目标英文最佳覆盖度位置:")
        print(f"    候选子句位置: {result['last_sentence']['en_best']['tail_length']}")
        print(f"    最佳匹配片段: '{result['last_sentence']['en_best']['segment']}'")
        print(f"    覆盖度: {result['last_sentence']['en_best']['coverage']:.3f}")
        print(f"    对应英文句子: '{Source_sentences_en_last}'")

        print(f"  目标中文最佳覆盖度位置:")
        print(f"    候选子句位置: {result['last_sentence']['zh_best']['tail_length']}")
        print(f"    最佳匹配片段: '{result['last_sentence']['zh_best']['segment']}'")
        print(f"    覆盖度: {result['last_sentence']['zh_best']['coverage']:.3f}")
        print(f"    对应中文句子: '{Source_sentences_zh_last}'")




if __name__ == "__main__":
    main()