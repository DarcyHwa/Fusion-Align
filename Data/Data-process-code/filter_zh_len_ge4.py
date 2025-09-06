# -*- coding: utf-8 -*-
"""
遍历 JSONL 文件，删除字段 "zh" 的值长度 < 4 的整行，并输出到新文件。
默认：
- 输入：Data-2/wmt23test.en-zh.jsonl（UTF-8）
- 输出：Data-2/wmt23test.en-zh.len_ge4.jsonl（UTF-8）
- 计数：对 zh 进行 strip() 后按 Unicode 字符数 len() 计算
- 缺少 zh 或 zh 非字符串：删除
- 无效 JSON 行：原样保留
- 空行：跳过且不输出
- 控制台打印统计信息

可在 Notebook 或命令行直接运行（纯标准库）。
"""
from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Dict

# 默认输入/输出路径（相对于仓库根目录）
INPUT_PATH = Path("dev-6/train_en-zh.jsonl")
OUTPUT_PATH = Path("dev-6/train_en-zh_4.jsonl")


def filter_jsonl_by_zh_length(
    input_path: Path,
    output_path: Path,
    min_len: int = 4,
    keep_invalid_json: bool = True,
    delete_if_missing_zh: bool = True,
) -> Dict[str, int]:
    """过滤 JSONL，删除 zh 字段长度小于 min_len 的行。

    返回统计：{"total": 总行数, "kept": 保留, "deleted": 删除,
             "invalid_json_kept": 无效JSON保留, "empty_skipped": 空行跳过,
             "missing_zh_deleted": 缺失/非字符串 zh 删除}
    """
    total = kept = deleted = 0
    invalid_json_kept = empty_skipped = missing_zh_deleted = 0

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="strict") as fin, \
            output_path.open("w", encoding="utf-8", newline="", errors="strict") as fout:
        for line in fin:
            total += 1
            if line.strip() == "":
                empty_skipped += 1
                continue
            try:
                obj = json.loads(line)
            except JSONDecodeError:
                if keep_invalid_json:
                    fout.write(line)
                    kept += 1
                    invalid_json_kept += 1
                else:
                    deleted += 1
                continue

            zh_val = obj.get("zh", None)
            if not isinstance(zh_val, str):
                if delete_if_missing_zh:
                    deleted += 1
                    missing_zh_deleted += 1
                    continue
                else:
                    fout.write(line)
                    kept += 1
                    continue

            # 以去除首尾空白后的字符数进行判断
            if len(zh_val.strip()) < min_len:
                deleted += 1
                continue

            # 保留原行文本写回，避免改变序列化格式
            fout.write(line)
            kept += 1

    return {
        "total": total,
        "kept": kept,
        "deleted": deleted,
        "invalid_json_kept": invalid_json_kept,
        "empty_skipped": empty_skipped,
        "missing_zh_deleted": missing_zh_deleted,
    }


def main() -> None:
    # 执行过滤并打印统计信息
    stats = filter_jsonl_by_zh_length(INPUT_PATH, OUTPUT_PATH, min_len=4,
                                      keep_invalid_json=True, delete_if_missing_zh=True)

    print("\n*** 过滤完成 ***")
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print("--- 统计信息 ---")
    print(f"总行数: {stats['total']}")
    print(f"保留行: {stats['kept']}")
    print(f"删除行: {stats['deleted']}")
    print(f"无效JSON保留: {stats['invalid_json_kept']}")
    print(f"空行跳过: {stats['empty_skipped']}")
    print(f"缺失/非字符串 zh 删除: {stats['missing_zh_deleted']}")


if __name__ == "__main__":
    main()

