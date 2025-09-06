# !pip install faster-whisper

from faster_whisper import WhisperModel
from datetime import timedelta
import json, os

# === 配置 ===
AUDIO_PATH = "audio.mp3"
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE = "float16"

# === 工具：时间戳格式 ===
def srt_ts(t: float) -> str:
    ms = int(round(float(t) * 1000))
    h, rem = divmod(ms // 1000, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms%1000:03d}"

def vtt_ts(t: float) -> str:
    ms = int(round(float(t) * 1000))
    h, rem = divmod(ms // 1000, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms%1000:03d}"

# === 加载模型并转录（不启用词级时间戳） ===
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
segments_gen, info = model.transcribe(
    AUDIO_PATH,
    beam_size=5,
    language="en",
    vad_filter=True,
    word_timestamps=False,   # ← 确保不返回词级时间戳（也可直接删掉这一行，默认即为 False）
)

# 转列表，便于打印与保存
segments = list(segments_gen)

# 控制台打印
for seg in segments:
    print("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))

# === 导出 SRT / VTT / JSON（无 words 字段） ===
base = "audio2subtitle"
srt_path = f"{base}.srt"
vtt_path = f"{base}.vtt"
json_path = f"{base}.json"

# SRT
with open(srt_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, 1):
        line = (seg.text or "").strip()
        if not line:
            continue
        f.write(f"{i}\n{srt_ts(seg.start)} --> {srt_ts(seg.end)}\n{line}\n\n")

# VTT
with open(vtt_path, "w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for seg in segments:
        line = (seg.text or "").strip()
        if not line:
            continue
        f.write(f"{vtt_ts(seg.start)} --> {vtt_ts(seg.end)}\n{line}\n\n")

# JSON（只含 start/end/text）
items = []
for seg in segments:
    line = (seg.text or "").strip()
    if not line:
        continue
    items.append({
        "start": float(seg.start),
        "end": float(seg.end),
        "text": line,
    })

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print("Saved:", srt_path, vtt_path, json_path)
