# 🚀 NVIDIA NIM API Proxy - Cloudflare Worker

将 **任意 NVIDIA NIM 免费模型** 部署到 Cloudflare Workers，提供 OpenAI 兼容的 API 端点。

## ✨ 特性

- 🔄 **OpenAI 兼容 API** - 无缝对接现有应用
- 🎯 **多模型支持** - 支持聊天、视觉、嵌入、图像生成等多种模型
- 📡 **流式输出** - 支持 SSE 流式响应
- 🔍 **动态模型列表** - 自动获取 NVIDIA NIM 最新模型
- 🌐 **CORS 支持** - 跨域访问友好
- ⚡ **边缘部署** - Cloudflare Workers 全球加速

---

## 📋 支持的模型

### 💬 聊天模型

| 模型别名 | NVIDIA 模型 ID | 说明 |
|---------|---------------|------|
| `llama-3.1-8b` | meta/llama-3.1-8b-instruct | Meta Llama 3.1 8B |
| `llama-3.1-70b` | meta/llama-3.1-70b-instruct | Meta Llama 3.1 70B |
| `llama-3.1-405b` | meta/llama-3.1-405b-instruct | Meta Llama 3.1 405B |
| `llama-3.2-1b` | meta/llama-3.2-1b-instruct | Meta Llama 3.2 1B |
| `llama-3.2-3b` | meta/llama-3.2-3b-instruct | Meta Llama 3.2 3B |
| `llama-3.3-70b` | meta/llama-3.3-70b-instruct | Meta Llama 3.3 70B |
| `mistral-large` | mistralai/mistral-large | Mistral Large |
| `mixtral-8x7b` | mistralai/mixtral-8x7b-instruct-v0.1 | Mixtral 8x7B |
| `mixtral-8x22b` | mistralai/mixtral-8x22b-instruct-v0.1 | Mixtral 8x22B |
| `mistral-7b` | mistralai/mistral-7b-instruct-v0.3 | Mistral 7B |
| `nemotron-70b` | nvidia/llama-3.1-nemotron-70b-instruct | NVIDIA Nemotron 70B |
| `nemotron-340b` | nvidia/nemotron-4-340b-instruct | NVIDIA Nemotron 340B |
| `gemma-2-2b` | google/gemma-2-2b-it | Google Gemma 2 2B |
| `gemma-2-9b` | google/gemma-2-9b-it | Google Gemma 2 9B |
| `gemma-2-27b` | google/gemma-2-27b-it | Google Gemma 2 27B |
| `phi-3-mini` | microsoft/phi-3-mini-4k-instruct | Microsoft Phi-3 Mini |
| `phi-3-medium` | microsoft/phi-3-medium-4k-instruct | Microsoft Phi-3 Medium |
| `phi-3.5-mini` | microsoft/phi-3.5-mini-instruct | Microsoft Phi-3.5 Mini |
| `glm-4-9b` | nvidia/glm-4-9b-chat | 智谱 GLM-4 9B |
| `glm-5-9b` | nvidia/glm-5-9b-chat | 智谱 GLM-5 9B |
| `qwen2.5-7b` | qwen/qwen2.5-7b-instruct | 阿里 Qwen 2.5 7B |
| `qwen2.5-72b` | qwen/qwen2.5-72b-instruct | 阿里 Qwen 2.5 72B |
| `deepseek-r1` | deepseek-ai/deepseek-r1 | DeepSeek R1 |
| `deepseek-v3` | deepseek-ai/deepseek-v3 | DeepSeek V3 |

### 👁️ 视觉模型

| 模型别名 | NVIDIA 模型 ID |
|---------|---------------|
| `llama-3.2-11b-vision` | meta/llama-3.2-11b-vision-instruct |
| `llama-3.2-90b-vision` | meta/llama-3.2-90b-vision-instruct |
| `phi-3-vision` | microsoft/phi-3-vision-128k-instruct |
| `neva-22b` | nvidia/neva-22b |
| `paligemma` | google/paligemma |
| `qwen2-vl-7b` | qwen/qwen2-vl-7b-instruct |

### 🎨 图像生成模型

| 模型别名 | NVIDIA 模型 ID |
|---------|---------------|
| `sd-3-medium` | stabilityai/stable-diffusion-3-medium |
| `sdxl` | stabilityai/stable-diffusion-xl-base-1.0 |
| `flux.1-dev` | black-forest-labs/flux.1-dev |
| `flux.1-schnell` | black-forest-labs/flux.1-schnell |

### 📊 嵌入模型

| 模型别名 | NVIDIA 模型 ID |
|---------|---------------|
| `nv-embedqa-e5` | nvidia/nv-embedqa-e5-v5 |
| `nv-embedqa-1b-v1` | nvidia/llama-3.2-nv-embedqa-1b-v1 |
| `e5-large-v2` | intfloat/e5-large-v2 |
| `bge-large` | baai/bge-large-en |

---

## 🔧 部署步骤

### 1. 克隆项目

```bash
git clone https://github.com/EtAorangE/cf-worker-glm5.git
cd cf-worker-glm5
bun install
```

### 2. 获取 NVIDIA API Key

1. 访问 [NVIDIA NIM](https://build.nvidia.com/)
2. 注册/登录账号
3. 在 API Keys 页面创建新的 Key
4. 免费用户可获得 5000 API 积分

### 3. 配置环境变量

创建 `.dev.vars` 文件：

```env
NVIDIA_API_KEY=nvapi-xxxxx
ENVIRONMENT=development
```

### 4. 登录 Cloudflare

```bash
bunx wrangler login
```

### 5. 设置生产环境密钥

```bash
bunx wrangler secret put NVIDIA_API_KEY
# 输入你的 NVIDIA API Key
```

### 6. 部署

```bash
bun run deploy
```

---

## 📡 API 使用说明

### 基础 URL

```
https://your-worker.your-subdomain.workers.dev
```

### 聊天补全

```bash
curl -X POST https://your-worker.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### 流式输出

```bash
curl -X POST https://your-worker.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

### 视觉模型

```bash
curl -X POST https://your-worker.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }]
  }'
```

### 嵌入向量

```bash
curl -X POST https://your-worker.workers.dev/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nv-embedqa-e5",
    "input": "这是一段需要生成嵌入向量的文本"
  }'
```

### 图像生成

```bash
curl -X POST https://your-worker.workers.dev/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "A beautiful sunset over the ocean",
    "n": 1,
    "size": "1024x1024"
  }'
```

### 获取模型列表

```bash
curl https://your-worker.workers.dev/v1/models
```

---

## 🔌 SDK 集成

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-worker.workers.dev/v1",
    api_key="not-needed"  # API Key 已在 Worker 中配置
)

response = client.chat.completions.create(
    model="llama-3.1-70b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://your-worker.workers.dev/v1",
    api_key="not-needed",
    model="llama-3.1-70b"
)

response = llm.invoke("Hello!")
print(response.content)
```

### JavaScript/TypeScript

```typescript
const response = await fetch('https://your-worker.workers.dev/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama-3.1-70b',
    messages: [{ role: 'user', content: 'Hello!' }],
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

---

## 📝 请求参数

### Chat Completions

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | string | 必填 | 模型别名或完整 NVIDIA 模型 ID |
| messages | array | 必填 | 对话消息数组 |
| temperature | number | 0.7 | 温度参数 (0-1) |
| max_tokens | number | 1024 | 最大生成 token 数 |
| top_p | number | 0.9 | Top-p 采样参数 |
| stream | boolean | false | 是否流式输出 |
| stop | array | - | 停止词列表 |

### Embeddings

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | string | 必填 | 嵌入模型别名 |
| input | string/array | 必填 | 输入文本 |
| encoding_format | string | float | 编码格式 |

### Image Generations

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | string | sdxl | 图像模型别名 |
| prompt | string | 必填 | 图像描述 |
| n | number | 1 | 生成数量 |
| size | string | 1024x1024 | 图像尺寸 |

---

## 🔐 安全建议

1. **API Key 保护** - 不要将 API Key 提交到代码仓库
2. **访问控制** - 可以添加认证中间件限制访问
3. **速率限制** - Cloudflare Workers 有内置的速率限制功能
4. **日志监控** - 使用 `wrangler tail` 监控请求

---

## 🆘 常见问题

### Q: 如何获取 NVIDIA API Key?

访问 [NVIDIA NIM](https://build.nvidia.com/)，注册账号后在 API Keys 页面创建。免费用户可获得 5000 API 积分。

### Q: 支持哪些模型？

支持 NVIDIA NIM 平台上的所有免费模型，包括 Llama、Mistral、Gemma、Phi、GLM、Qwen、DeepSeek 等系列。

### Q: 可以使用完整的 NVIDIA 模型 ID 吗？

可以！除了使用别名，你也可以直接使用完整的 NVIDIA 模型 ID，如 `meta/llama-3.1-70b-instruct`。

### Q: 如何查看日志？

```bash
bun run tail
```

### Q: 本地开发时如何测试？

```bash
bun run dev
# 访问 http://localhost:8787
```

---

## 📄 License

MIT
