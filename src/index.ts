/**
 * Cloudflare Worker - NVIDIA GLM-5 API 代理
 *
 * 将 NVIDIA NIM GLM-5 模型部署为 OpenAI 兼容的 API 端点
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { streamSSE } from 'hono/streaming';

// 环境变量类型定义
interface Env {
  NVIDIA_API_KEY: string;
  ENVIRONMENT: string;
}

// OpenAI 兼容的请求类型
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  stream?: boolean;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop?: string[];
}

// NVIDIA NIM API 响应类型
interface NVIDIAChoice {
  index: number;
  message: {
    role: string;
    content: string;
  };
  finish_reason: string;
}

interface NVIDIAUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

interface NVIDIAResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: NVIDIAChoice[];
  usage: NVIDIAUsage;
}

// 创建 Hono 应用
const app = new Hono<{ Bindings: Env }>();

// 中间件
app.use('*', logger());
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

// 健康检查端点
app.get('/', (c) => {
  return c.json({
    status: 'ok',
    service: 'GLM-5 API Proxy',
    version: '1.0.0',
    endpoints: {
      chat: '/v1/chat/completions',
      models: '/v1/models',
      health: '/'
    }
  });
});

// 模型列表端点
app.get('/v1/models', (c) => {
  return c.json({
    object: 'list',
    data: [
      {
        id: 'glm-5',
        object: 'model',
        created: 1700000000,
        owned_by: 'nvidia',
        permission: [],
        root: 'glm-5',
        parent: null,
      },
      {
        id: 'glm-5-9b-chat',
        object: 'model',
        created: 1700000000,
        owned_by: 'nvidia',
        permission: [],
        root: 'glm-5-9b-chat',
        parent: null,
      },
      {
        id: 'glm-5-9b-chat-4k',
        object: 'model',
        created: 1700000000,
        owned_by: 'nvidia',
        permission: [],
        root: 'glm-5-9b-chat-4k',
        parent: null,
      }
    ]
  });
});

// 聊天补全端点
app.post('/v1/chat/completions', async (c) => {
  const apiKey = c.env.NVIDIA_API_KEY;

  if (!apiKey) {
    return c.json({
      error: {
        message: 'NVIDIA API Key not configured',
        type: 'configuration_error',
        code: 'missing_api_key'
      }
    }, 500);
  }

  try {
    const body = await c.req.json<ChatCompletionRequest>();

    // 验证请求
    if (!body.messages || body.messages.length === 0) {
      return c.json({
        error: {
          message: 'messages is required and cannot be empty',
          type: 'invalid_request_error',
          code: 'invalid_messages'
        }
      }, 400);
    }

    // 映射模型名称到 NVIDIA NIM 模型 ID
    const modelMapping: Record<string, string> = {
      'glm-5': 'nvidia/glm-5-9b-chat',
      'glm-5-9b-chat': 'nvidia/glm-5-9b-chat',
      'glm-5-9b-chat-4k': 'nvidia/glm-5-9b-chat-4k',
    };

    const nvidiaModel = modelMapping[body.model] || 'nvidia/glm-5-9b-chat';

    // 构建 NVIDIA NIM API 请求
    const nvidiaRequest = {
      model: nvidiaModel,
      messages: body.messages,
      temperature: body.temperature ?? 0.7,
      max_tokens: body.max_tokens ?? 1024,
      top_p: body.top_p ?? 0.9,
      stream: body.stream ?? false,
      frequency_penalty: body.frequency_penalty,
      presence_penalty: body.presence_penalty,
      stop: body.stop,
    };

    // 处理流式响应
    if (body.stream) {
      return streamSSE(c, async (stream) => {
        const response = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
            'Accept': 'text/event-stream',
          },
          body: JSON.stringify(nvidiaRequest),
        });

        if (!response.ok) {
          const errorText = await response.text();
          await stream.writeSSE({
            event: 'error',
            data: JSON.stringify({
              error: {
                message: `NVIDIA API error: ${response.status}`,
                details: errorText,
              }
            }),
          });
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          await stream.writeSSE({
            event: 'error',
            data: JSON.stringify({ error: 'No response body' }),
          });
          return;
        }

        const decoder = new TextDecoder();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                  await stream.writeSSE({
                    event: 'message',
                    data: '[DONE]',
                  });
                } else {
                  await stream.writeSSE({
                    event: 'message',
                    data: data,
                  });
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      });
    }

    // 非流式响应
    const response = await fetch('https://integrate.api.nvidia.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify(nvidiaRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('NVIDIA API Error:', response.status, errorText);

      return c.json({
        error: {
          message: `NVIDIA API error: ${response.status}`,
          type: 'api_error',
          code: 'nvidia_api_error',
          details: errorText,
        }
      }, response.status as 400 | 401 | 403 | 404 | 429 | 500 | 502 | 503);
    }

    const data = await response.json<NVIDIAResponse>();

    // 转换响应格式以保持 OpenAI 兼容性
    return c.json({
      id: data.id,
      object: data.object,
      created: data.created,
      model: body.model,
      choices: data.choices.map(choice => ({
        index: choice.index,
        message: {
          role: choice.message.role,
          content: choice.message.content,
        },
        finish_reason: choice.finish_reason,
      })),
      usage: data.usage,
    });

  } catch (error) {
    console.error('Request processing error:', error);

    return c.json({
      error: {
        message: error instanceof Error ? error.message : 'Internal server error',
        type: 'internal_error',
        code: 'internal_error',
      }
    }, 500);
  }
});

// 错误处理
app.notFound((c) => {
  return c.json({
    error: {
      message: 'Not found',
      type: 'not_found',
      code: 'not_found',
    }
  }, 404);
});

app.onError((err, c) => {
  console.error('Unhandled error:', err);
  return c.json({
    error: {
      message: err.message,
      type: 'internal_error',
      code: 'internal_error',
    }
  }, 500);
});

// 导出 Worker
export default app;
