import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-V3",
    "thinking_budget": 4096,
    "top_p": 0.5,
    "messages": [
        {
            "content": "你是一个翻译器,把用户发送的英文翻译成中文. 遵循:翻译后的中文句子要与原语言的英文句子保持相同的语义和语序顺序.",
            "role": "system"
        },
        {
            "content": "Now, that might sound like a mouthful,But basically, the reason why we call this quadratic funding is that it leverages squares and square roots in order to determine the matching amounts.g",
            "role": "user"
        }
    ],
    "enable_thinking": False,
    "stream": False
}
headers = {
    "Authorization": "Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())