# -*- coding: utf-8 -*-
"""
在 Notebook 环境下可直接运行的独立脚本：
- 遍历 JSONL 文件（每行一个 JSON 或任意文本行）
- 如果某一整行文本中 **出现以下任一标记**： "["、"<"、"–"(en-dash, U+2013)、"..."（三个点）
  则 **删除该行**（不写入输出）
- 其它行原样写入输出（不解析/不改动内容）
- 保留空行
- 逐行流式处理，适合大文件
- 打印处理摘要

使用方法：
1) 在“配置区”把 IN_PATH / OUT_PATH 改成你的路径
2) 运行本单元格
"""

from __future__ import annotations
import os
from typing import List, Dict

# =========================
# 配置区（请修改为你的路径）
# =========================
IN_PATH  = "dev-6/train_en-zh.jsonl"            # 输入 JSONL
OUT_PATH = "dev-6/train_en-zh2.jsonl"  # 输出 JSONL

# 要匹配的标记（任意一个出现即删除该行）
TOKENS: List[str] = ["[", "<", "–", "..."]  # 注意 "–" 是 en-dash（不是普通连字符）

# 是否区分大小写：对这些标记而言无意义，这里保留开关以便扩展
CASE_SENSITIVE = True

# =========================
# 核心逻辑
# =========================
def _contains_any(haystack: str, needles: List[str], case_sensitive: bool = True) -> bool:
    """只基于整行原始文本做子串查找，不做 JSON 解析。"""
    if not case_sensitive:
        haystack = haystack.lower()
        needles = [n.lower() for n in needles]
    return any(n in haystack for n in needles)

def filter_lines_by_tokens(
    in_path: str,
    out_path: str,
    tokens: List[str],
    case_sensitive: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, int]:
    """
    逐行读取 in_path；若某行文本包含任一 token，则跳过；否则写入 out_path。
    返回处理统计。
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"输入文件不存在：{in_path}")

    total = 0
    kept = 0
    dropped = 0
    blank_lines = 0

    with open(in_path, "r", encoding=encoding, errors="strict") as fin, \
         open(out_path, "w", encoding=encoding, errors="strict") as fout:

        for line in fin:
            # 保留原始换行：我们只去掉末尾 '\n' 用于判断，写出时使用原行或加回 '\n'
            raw = line.rstrip("\n")
            if raw == "":
                # 保留空行
                fout.write("\n")
                blank_lines += 1
                continue

            total += 1

            if _contains_any(raw, tokens, case_sensitive):
                dropped += 1
                continue

            # 未命中则原样写出（保持非 ASCII 字符）
            fout.write(raw + "\n")
            kept += 1

    summary = {
        "input": in_path,
        "output": out_path,
        "total_nonblank_lines_read": total,
        "blank_lines_preserved": blank_lines,
        "kept_lines": kept,
        "dropped_lines": dropped,
        "tokens": len(tokens),
    }
    return summary

# =========================
# 执行
# =========================
stats = filter_lines_by_tokens(
    IN_PATH,
    OUT_PATH,
    tokens=TOKENS,
    case_sensitive=CASE_SENSITIVE,
    encoding="utf-8",
)

print("FILTER SUMMARY")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n完成。输出文件：", OUT_PATH)
