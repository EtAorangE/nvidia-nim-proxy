# 🚀 Cloudflare Worker - NVIDIA GLM-5 部署

这是一个将 NVIDIA GLM-5 模型部署到 Cloudflare Workers 的项目。

## 📋 前置要求

1. **Cloudflare 账号** - 需要一个 Cloudflare 账号
2. **NVIDIA API Key** - 从 [NVIDIA NIM](https://build.nvidia.com/) 获取 API Key
3. **Node.js/Bun** - 本地开发环境

## 🔧 配置步骤

### 1. 安装依赖

```bash
cd cf-worker-glm5
bun install
# 或
npm install
```

### 2. 配置环境变量

在 Cloudflare Dashboard 中设置以下环境变量，或创建 `.dev.vars` 文件用于本地开发：

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. 登录 Cloudflare

```bash
bunx wrangler login
```

### 4. 本地开发

```bash
bun run dev
```

### 5. 部署到 Cloudflare

```bash
bun run deploy
```

## 📡 API 使用说明

### 端点

部署后，你的 Worker 将提供以下端点：

```
POST https://your-worker.your-subdomain.workers.dev/v1/chat/completions
```

### 请求示例

```bash
curl -X POST https://your-worker.your-subdomain.workers.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| messages | array | 必填 | 对话消息数组 |
| temperature | number | 0.7 | 温度参数 (0-1) |
| max_tokens | number | 1024 | 最大生成 token 数 |
| top_p | number | 0.9 | Top-p 采样参数 |
| stream | boolean | false | 是否流式输出 |

### 响应格式

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "glm-5",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

## 🔐 安全建议

1. **API Key 保护** - 不要将 API Key 提交到代码仓库
2. **访问控制** - 可以添加认证中间件限制访问
3. **速率限制** - Cloudflare Workers 有内置的速率限制功能

## 📝 自定义配置

修改 `wrangler.toml` 文件可以自定义：

- Worker 名称
- 兼容性日期
- 环境变量绑定
- KV 存储绑定

## 🆘 常见问题

### Q: 如何获取 NVIDIA API Key?

访问 [NVIDIA NIM](https://build.nvidia.com/)，注册账号后在 API Keys 页面创建。

### Q: 支持哪些 GLM-5 模型?

目前支持 NVIDIA NIM 平台上的 GLM-5 系列模型，包括：
- glm-5-9b-chat
- glm-5-9b-chat-4k

### Q: 如何查看日志?

```bash
bun run tail
```

## 📄 License

MIT
