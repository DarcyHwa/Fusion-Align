import requests
import numpy as np
from transformers import AutoTokenizer
import unicodedata
from scipy.spatial.distance import cosine

# ========== 全局配置 ==========
# XLM-RoBERTa-XL 模型配置（仅用于分词）
MODEL_ID = "facebook/xlm-roberta-xl"

# Qwen3-Embedding-8B API 配置
API_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 注：现在只使用分词功能，不需要GPU计算

# 加载 XLM-RoBERTa-XL 分词器（仅用于分词）
print(f"正在加载分词器: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"分词器加载完成")


def get_local_tokenization(text):
    """使用本地 XLM-RoBERTa-XL 模型进行分词（不包含特殊标记）"""
    # 设置 add_special_tokens=False 确保不包含特殊标记
    inputs = tokenizer(text, padding=False, truncation=True,
                      max_length=512, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])

    # 解码token为可读文本，过滤掉空的或无效的token
    filtered_tokens = []
    for token in tokens:
        decoded = decode_token(token)
        # 只添加非空的、有效的token
        if decoded is not None and decoded.strip() != "":
            filtered_tokens.append(decoded)

    return filtered_tokens


def is_special_token(token):
    """判断是否为特殊标记"""
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


def decode_token(token):
    """解码单个token为可读文本，移除SentencePiece特殊标记"""
    try:
        # 先使用tokenizer解码
        s = tokenizer.convert_tokens_to_string([token]).strip()
        
        # 移除SentencePiece特殊标记
        # '▁' 是SentencePiece中表示词开始的标记
        s = s.replace('▁', '')
        
        # 移除其他常见的特殊标记
        special_marks = ['<s>', '</s>', '<pad>', '<unk>', '<mask>']
        for mark in special_marks:
            s = s.replace(mark, '')
        
        # 去除前后空白
        s = s.strip()
        
        # 如果解码后为空字符串，返回None表示应该过滤掉这个token
        return s if s != "" else None
    except Exception:
        # 如果解码失败，检查原始token是否包含特殊标记
        if '▁' in token:
            cleaned_token = token.replace('▁', '').strip()
            return cleaned_token if cleaned_token != "" else None
        return None


def get_qwen_embeddings(text_list):
    """调用 Qwen3-Embedding-8B 模型获取嵌入向量"""
    if not text_list:
        return []

    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text_list,
        "encoding_format": "float"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers, timeout=60)

    try:
        resp_json = response.json()
        embeddings = [item.get("embedding") for item in resp_json.get("data", [])]
    except ValueError:
        raise RuntimeError(f"接口返回非 JSON 内容: {response.text[:200]}")

    if len(embeddings) != len(text_list):
        raise RuntimeError("返回的嵌入数量与输入文本数量不一致")

    # 转换为与原代码兼容的格式
    class EmbeddingObject:
        def __init__(self, values):
            self.values = values

    return [EmbeddingObject(emb) for emb in embeddings]





def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    if not vec1 or not vec2:
        return 0

    # 处理numpy数组和列表
    if hasattr(vec1, 'flatten'):
        vec1 = vec1.flatten()
    if hasattr(vec2, 'flatten'):
        vec2 = vec2.flatten()

    return 1 - cosine(vec1, vec2)


def calculate_similarity_matrix(embeddings1, embeddings2):
    """计算两组嵌入向量之间的余弦相似度矩阵"""
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))

    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            if hasattr(emb1, 'values'):
                vec1 = emb1.values
            else:
                vec1 = emb1

            if hasattr(emb2, 'values'):
                vec2 = emb2.values
            else:
                vec2 = emb2

            similarity_matrix[i, j] = cosine_similarity(vec1, vec2)

    return similarity_matrix








def is_punct_text(text):
    """判断文本是否全为标点符号"""
    if not text:
        return False
    for ch in text:
        cat = unicodedata.category(ch)
        if not (cat.startswith('P') or cat.startswith('S')):
            return False
    return True








def find_optimal_split_semantic(result, forward_candidates, backward_candidates,
                               forward_english_sim, backward_english_sim):
    """
    找到目标句子分割的最优分割点（基于语义相似度）

    参数:
        result: 目标句子分割分词后的文本
        forward_candidates: 目标句子候选子句(前向)列表
        backward_candidates: 目标句子候选子句(后向)列表
        forward_english_sim: 目标句子候选子句(前向)与源英文第一句的语义相似度矩阵
        backward_english_sim: 目标句子候选子句(后向)与源英文最后一句的语义相似度矩阵

    返回:
        元组 (split_index, forward_index, backward_index, scores)
        其中scores是一个包含各项得分的字典
    """
    n = len(result)
    best_total_score = -1
    best_split = None
    best_scores = None

    # 尝试所有可能的分割点
    for split_idx in range(1, n):
        # 获取当前分割点的前向和后向部分
        forward_tokens = result[:split_idx]
        backward_tokens = result[split_idx:]

        # 如果任一部分太短（少于2个token），则跳过
        if len(forward_tokens) < 2 or len(backward_tokens) < 2:
            continue

        # 找到对应的候选索引
        forward_idx = len(forward_tokens) - 2  # 目标句子候选子句(前向)从长度2开始，0索引
        backward_idx = len(backward_tokens) - 2  # 目标句子候选子句(后向)从长度2开始，0索引

        # 如果索引超出范围，则跳过
        if (forward_idx < 0 or forward_idx >= len(forward_english_sim) or
                backward_idx < 0 or backward_idx >= len(backward_english_sim)):
            continue

        # 计算语义相似度得分
        forward_semantic = forward_english_sim[forward_idx, 0]
        backward_semantic = backward_english_sim[backward_idx, 0]

        # 计算总得分（纯语义相似度）
        total_score = forward_semantic + backward_semantic

        # 如果得分更高，则更新最佳分割点
        if total_score > best_total_score:
            best_total_score = total_score
            best_split = (split_idx, forward_idx, backward_idx)
            best_scores = {
                'forward_semantic': forward_semantic,
                'backward_semantic': backward_semantic,
                'total_score': total_score
            }

    if best_split:
        return best_split[0], best_split[1], best_split[2], best_scores
    else:
        return None, None, None, None


def print_embeddings(embeddings, text_list, type_name):
    """以格式化方式打印嵌入向量"""
    print(f"\n{type_name}的嵌入向量:")

    for i, embedding in enumerate(embeddings):
        print(f"{type_name} {i + 1} '{text_list[i]}' 的嵌入向量:")
        print(f"向量维度: {len(embedding.values)}")
        print(f"前10个值: {embedding.values[:10]}")
        print("-" * 50)


def print_similarity_matrix(matrix, row_labels, col_labels, name):
    """打印相似度矩阵"""
    print(f"\n{name}相似度矩阵:")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            print(f"{row_labels[i]} 与 {col_labels[j]} 的相似度: {val:.4f}")


def handle_middle_sentences(Source_sentences_en, chinese_text, best_forward_text, best_backward_text):
    """处理中间源英文句子与剩余目标句子分割字符的对应逻辑"""
    if len(Source_sentences_en) <= 2:
        print("源英文句子数量 <= 2，无需处理中间句子")
        return None

    print(f"\n===== 处理中间源英文句子 =====")

    # 计算剩余的目标句子分割字符串
    chinese_tokens = get_local_tokenization(chinese_text)
    forward_tokens = get_local_tokenization(best_forward_text)
    backward_tokens = get_local_tokenization(best_backward_text)

    # 找到前向和后向部分在原目标句子分割中的位置
    forward_len = len(forward_tokens)
    backward_len = len(backward_tokens)

    if forward_len + backward_len < len(chinese_tokens):
        # 计算剩余的中间部分
        middle_tokens = chinese_tokens[forward_len:len(chinese_tokens) - backward_len]
        middle_text = ''.join(middle_tokens)

        print(f"剩余的中间目标句子分割字符: '{middle_text}'")

        # 处理中间的源英文句子（除了第一个和最后一个）
        middle_Source_sentences_en = Source_sentences_en[1:-1]
        print(f"中间的源英文句子: {middle_Source_sentences_en}")

        # 计算中间源英文句子与中间目标句子分割字符的相似度
        results = []
        for i, eng_clause in enumerate(middle_Source_sentences_en):
            semantic_sim_emb = get_qwen_embeddings([eng_clause, middle_text])
            semantic_similarity = cosine_similarity(semantic_sim_emb[0].values, semantic_sim_emb[1].values)

            results.append({
                'english_clause': eng_clause,
                'semantic_similarity': semantic_similarity
            })

            print(f"源英文中间句子 {i + 1}: '{eng_clause}'")
            print(f"  语义相似度: {semantic_similarity:.4f}")

        return {
            'middle_chinese_text': middle_text,
            'middle_english_results': results
        }
    else:
        print("前向和后向部分已覆盖整个目标句子分割字符串，无中间部分")
        return None



def analyze_tokenization_process():
    """分析token处理流程"""
    print("=" * 70)
    print("📖 token分词处理流程")
    print("=" * 70)
    
    test_text = "the everyday citizens in these communities"
    print(f"📝 分析文本: '{test_text}'")
    print("-" * 70)
    
    # token分词过程
    print("🔄 token分词过程:")
    tokens = get_local_tokenization(test_text)
    
    print(f"   1. 模型分词器处理文本")
    print(f"   2. 转换为token字符串（不含特殊标记）")
    print(f"   3. 解码token为可读文本")
    
    print(f"\n🔤 最终token结果:")
    print(f"   数量: {len(tokens)} 个token")
    print(f"   内容: {' | '.join(tokens)}")
    
    # 显示每个token的解码过程
    print(f"\n📋 token解码详情:")
    inputs = tokenizer(test_text, padding=False,
                      truncation=True, max_length=512, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])
    
    for i, token in enumerate(raw_tokens):
        decoded = decode_token(token)
        print(f"   [{i+1}] '{token}' → '{decoded}'")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 60)
    print("开始主程序")
    print("=" * 60)

    Source_sentences_en = [
        "in which the everyday citizens in these communities",
        "contribute to the projects that are in the campaign"
    ]

    Source_sentences_zh = [
        "这些社区的普通居民",
        "为活动中的项目捐款"
    ]

    Target_sentence_segmentation = "由社区里的普通市民为这些活动中的项目出资"

    # 使用本地 XLM-RoBERTa-XL 模型进行目标句子分割分词
    result = get_local_tokenization(Target_sentence_segmentation)
    print(f"🀄 目标句子分割: '{Target_sentence_segmentation}'")
    print(f"🔤 [token分词] 目标句子分割分词: {' | '.join(result)}")
    print(f"   Token数量: {len(result)}")
    print('=' * 70)

    # 创建从前向后的目标句子候选子句
    forward_candidates = []
    for i in range(2, len(result) + 1):
        sub_clause_list = result[0:i]
        sub_clause_str = ''.join(sub_clause_list)
        forward_candidates.append(sub_clause_str)

    # 创建从后向前的目标句子候选子句
    backward_candidates = []
    for i in range(2, len(result) + 1):
        sub_clause_list = result[-i:]
        sub_clause_str = ''.join(sub_clause_list)
        backward_candidates.append(sub_clause_str)

    # 打印目标句子候选子句信息
    print(f"\n从前向后总共生成了 {len(forward_candidates)} 个目标句子候选子句")
    for i, candidate in enumerate(forward_candidates):
        print(f"目标句子候选子句(前向) {i + 1}: {candidate}")

    print(f"\n从后向前总共生成了 {len(backward_candidates)} 个目标句子候选子句")
    for i, candidate in enumerate(backward_candidates):
        print(f"目标句子候选子句(后向) {i + 1}: {candidate}")

    print("\n开始处理嵌入向量...")

    # 使用 Qwen3-Embedding-8B 模型嵌入源英文句子
    print("\n处理源英文句子的嵌入向量...")
    english_embeddings = get_qwen_embeddings(Source_sentences_en)
    print_embeddings(english_embeddings, Source_sentences_en, "源英文句子")

    # 嵌入目标句子候选子句(前向)
    print("\n处理目标句子候选子句(前向)的嵌入向量...")
    forward_embeddings = get_qwen_embeddings(forward_candidates)
    print_embeddings(forward_embeddings, forward_candidates, "目标句子候选子句(前向)")

    # 嵌入目标句子候选子句(后向)
    print("\n处理目标句子候选子句(后向)的嵌入向量...")
    backward_embeddings = get_qwen_embeddings(backward_candidates)
    print_embeddings(backward_embeddings, backward_candidates, "目标句子候选子句(后向)")

    # ===== 前向处理 =====
    print("\n===== 目标句子候选子句(前向)处理 =====")

    # 目标句子候选子句(前向)与源英文第一句的句子嵌入相似度
    print("\n--- 目标句子候选子句(前向)与源英文第一句 ---")

    # 计算目标句子候选子句(前向)与源英文第一句的语义相似度矩阵
    print("\n计算语义相似度矩阵...")
    print("使用Qwen3-Embedding-8B模型计算目标句子候选子句(前向)与源英文第一句之间的余弦相似度")
    forward_english_sim = calculate_similarity_matrix(forward_embeddings, [english_embeddings[0]])

    print("\n💯 计算句子嵌入相似度...")
    for i, candidate in enumerate(forward_candidates):
        similarity = forward_english_sim[i, 0]
        print(f"📊 目标句子候选子句(前向){i + 1} → 源英文第一句")
        print(f"   📈 嵌入相似度: {similarity:.4f}")
        print(f"   💬 '{candidate}' ↔ '{Source_sentences_en[0]}'")

    # 目标句子候选子句(前向)与源中文第一句的句子嵌入相似度
    print("\n--- 目标句子候选子句(前向)与源中文第一句 ---")
    chinese_first_embedding = get_qwen_embeddings([Source_sentences_zh[0]])
    forward_chinese_sim = calculate_similarity_matrix(forward_embeddings, chinese_first_embedding)

    print("\n💯 计算句子嵌入相似度...")
    for i, candidate in enumerate(forward_candidates):
        similarity = forward_chinese_sim[i, 0]
        print(f"📊 目标句子候选子句(前向){i + 1} → 源中文第一句")
        print(f"   📈 嵌入相似度: {similarity:.4f}")
        print(f"   💬 '{candidate}' ↔ '{Source_sentences_zh[0]}'")

    # ===== 后向处理 =====
    print("\n===== 目标句子候选子句(后向)处理 =====")

    # 目标句子候选子句(后向)与源英文最后一句的句子嵌入相似度
    print("\n--- 目标句子候选子句(后向)与源英文最后一句 ---")
    backward_english_sim = calculate_similarity_matrix(backward_embeddings, [english_embeddings[-1]])

    print("\n💯 计算句子嵌入相似度...")
    for i, candidate in enumerate(backward_candidates):
        similarity = backward_english_sim[i, 0]
        print(f"📊 目标句子候选子句(后向){i + 1} → 源英文最后一句")
        print(f"   📈 嵌入相似度: {similarity:.4f}")
        print(f"   💬 '{candidate}' ↔ '{Source_sentences_en[-1]}'")

    # 目标句子候选子句(后向)与源中文最后一句的句子嵌入相似度
    print("\n--- 目标句子候选子句(后向)与源中文最后一句 ---")
    chinese_last_embedding = get_qwen_embeddings([Source_sentences_zh[-1]])
    backward_chinese_sim = calculate_similarity_matrix(backward_embeddings, chinese_last_embedding)

    print("\n💯 计算句子嵌入相似度...")
    for i, candidate in enumerate(backward_candidates):
        similarity = backward_chinese_sim[i, 0]
        print(f"📊 目标句子候选子句(后向){i + 1} → 源中文最后一句")
        print(f"   📈 嵌入相似度: {similarity:.4f}")
        print(f"   💬 '{candidate}' ↔ '{Source_sentences_zh[-1]}'")


if __name__ == "__main__":
    main()