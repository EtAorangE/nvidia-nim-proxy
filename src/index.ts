/**
 * Cloudflare Worker - NVIDIA NIM API 通用代理
 *
 * 支持部署 NVIDIA NIM 平台上的所有免费模型到 Cloudflare Workers
 * 提供 OpenAI 兼容的 API 端点
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { streamSSE } from 'hono/streaming';

// 环境变量类型定义
interface Env {
  NVIDIA_API_KEY: string;
  ENVIRONMENT: string;
  DEFAULT_MODEL?: string;
}

// OpenAI 兼容的请求类型
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
}

interface ContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
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
  n?: number;
}

// 图像生成请求类型
interface ImageGenerationRequest {
  prompt: string;
  model?: string;
  n?: number;
  size?: string;
  quality?: string;
  style?: string;
  response_format?: 'url' | 'b64_json';
}

// 嵌入请求类型
interface EmbeddingRequest {
  model: string;
  input: string | string[];
  encoding_format?: 'float' | 'base64';
}

// NVIDIA NIM 支持的模型配置
const NVIDIA_MODELS = {
  // 聊天/文本模型
  chat: {
    // Meta Llama 系列
    'llama-3.1-8b': 'meta/llama-3.1-8b-instruct',
    'llama-3.1-70b': 'meta/llama-3.1-70b-instruct',
    'llama-3.1-405b': 'meta/llama-3.1-405b-instruct',
    'llama-3.2-1b': 'meta/llama-3.2-1b-instruct',
    'llama-3.2-3b': 'meta/llama-3.2-3b-instruct',
    'llama-3.3-70b': 'meta/llama-3.3-70b-instruct',

    // Mistral 系列
    'mistral-large': 'mistralai/mistral-large',
    'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct-v0.1',
    'mixtral-8x22b': 'mistralai/mixtral-8x22b-instruct-v0.1',
    'mistral-7b': 'mistralai/mistral-7b-instruct-v0.3',

    // NVIDIA Nemotron 系列
    'nemotron-70b': 'nvidia/llama-3.1-nemotron-70b-instruct',
    'nemotron-340b': 'nvidia/nemotron-4-340b-instruct',

    // Google Gemma 系列
    'gemma-2-2b': 'google/gemma-2-2b-it',
    'gemma-2-9b': 'google/gemma-2-9b-it',
    'gemma-2-27b': 'google/gemma-2-27b-it',
    'recurrentgemma-2b': 'google/recurrentgemma-2b-it',

    // Microsoft Phi 系列
    'phi-3-mini': 'microsoft/phi-3-mini-4k-instruct',
    'phi-3-mini-128k': 'microsoft/phi-3-mini-128k-instruct',
    'phi-3-medium': 'microsoft/phi-3-medium-4k-instruct',
    'phi-3-medium-128k': 'microsoft/phi-3-medium-128k-instruct',
    'phi-3.5-mini': 'microsoft/phi-3.5-mini-instruct',

    // 智谱 GLM 系列
    'glm-4-9b': 'nvidia/glm-4-9b-chat',
    'glm-5-9b': 'nvidia/glm-5-9b-chat',

    // 阿里 Qwen 系列
    'qwen2.5-7b': 'qwen/qwen2.5-7b-instruct',
    'qwen2.5-72b': 'qwen/qwen2.5-72b-instruct',
    'qwen2-7b': 'qwen/qwen2-7b-instruct',
    'qwen2-72b': 'qwen/qwen2-72b-instruct',

    // DeepSeek 系列
    'deepseek-r1': 'deepseek-ai/deepseek-r1',
    'deepseek-v3': 'deepseek-ai/deepseek-v3',

    // 其他模型
    'arctic': 'snowflake/arctic',
    'granite-3.0-8b': 'ibm/granite-3.0-8b-instruct',
    'granite-3.1-8b': 'ibm/granite-3.1-8b-instruct',
    'starcoder2-7b': 'bigcode/starcoder2-7b',
    'starcoder2-15b': 'bigcode/starcoder2-15b',
  },

  // 视觉模型
  vision: {
    'llama-3.2-11b-vision': 'meta/llama-3.2-11b-vision-instruct',
    'llama-3.2-90b-vision': 'meta/llama-3.2-90b-vision-instruct',
    'phi-3-vision': 'microsoft/phi-3-vision-128k-instruct',
    'neva-22b': 'nvidia/neva-22b',
    'paligemma': 'google/paligemma',
    'fuyu-8b': 'adept/fuyu-8b',
    'kosmos-2': 'microsoft/kosmos-2',
    'vila': 'nvidia/vila',
    'qwen2-vl-7b': 'qwen/qwen2-vl-7b-instruct',
  },

  // 图像生成模型
  image: {
    'sd-3-medium': 'stabilityai/stable-diffusion-3-medium',
    'sdxl': 'stabilityai/stable-diffusion-xl-base-1.0',
    'sd-1.5': 'runwayml/stable-diffusion-v1-5',
    'flux.1-dev': 'black-forest-labs/flux.1-dev',
    'flux.1-schnell': 'black-forest-labs/flux.1-schnell',
  },

  // 嵌入模型
  embedding: {
    'nv-embedqa-e5': 'nvidia/nv-embedqa-e5-v5',
    'nv-embedqa-1b-v1': 'nvidia/llama-3.2-nv-embedqa-1b-v1',
    'nv-embedqa-1b-v2': 'nvidia/llama-3.2-nv-embedqa-1b-v2',
    'e5-large-v2': 'intfloat/e5-large-v2',
    'bge-large': 'baai/bge-large-en',
    'snowflake-arctic-embed': 'snowflake/arctic-embed-l',
  },

  // 视频生成模型
  video: {
    'gen-3-alpha-turbo': 'runway/gen-3-alpha-turbo',
  },

  // 语音识别模型
  asr: {
    'parakeet-tdt': 'nvidia/parakeet-tdt-0.6b-v2',
    'canary-1b': 'nvidia/canary-1b',
    'whisper-large-v3': 'openai/whisper-large-v3',
  },

  // 语音合成模型
  tts: {
    'radtts-hifigan': 'nvidia/radtts-hifigan-r2',
  },

  // 重排序模型
  rerank: {
    'nvidia-rerankqa': 'nvidia/nv-rerankqa-mistral-4b-v3',
    'cohere-rerank': 'cohere/rerank-english-v3.0',
  }
};

// 获取 NVIDIA 模型 ID
function getNvidiaModelId(modelAlias: string): string {
  // 检查是否是完整的 NVIDIA 模型 ID
  if (modelAlias.includes('/')) {
    return modelAlias;
  }

  // 查找模型别名
  for (const category of Object.values(NVIDIA_MODELS)) {
    if (modelAlias in category) {
      return category[modelAlias as keyof typeof category];
    }
  }

  // 默认返回
  return `nvidia/${modelAlias}`;
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
    service: 'NVIDIA NIM API Proxy',
    version: '2.0.0',
    description: 'Deploy any NVIDIA NIM model to Cloudflare Workers',
    endpoints: {
      chat: '/v1/chat/completions',
      models: '/v1/models',
      embeddings: '/v1/embeddings',
      images: '/v1/images/generations',
      health: '/'
    },
    supported_models: {
      chat: Object.keys(NVIDIA_MODELS.chat).length + ' models',
      vision: Object.keys(NVIDIA_MODELS.vision).length + ' models',
      image: Object.keys(NVIDIA_MODELS.image).length + ' models',
      embedding: Object.keys(NVIDIA_MODELS.embedding).length + ' models',
    }
  });
});

// 模型列表端点
app.get('/v1/models', async (c) => {
  const apiKey = c.env.NVIDIA_API_KEY;

  // 尝试从 NVIDIA API 获取实时模型列表
  let liveModels: any[] = [];

  if (apiKey) {
    try {
      const response = await fetch('https://integrate.api.nvidia.com/v1/models', {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
        },
      });

      if (response.ok) {
        const data = await response.json() as { data: any[] };
        liveModels = data.data || [];
      }
    } catch (error) {
      console.error('Failed to fetch live models:', error);
    }
  }

  // 构建本地模型列表
  const localModels: any[] = [];

  for (const [alias, nvidiaId] of Object.entries(NVIDIA_MODELS.chat)) {
    localModels.push({
      id: alias,
      object: 'model',
      created: 1700000000,
      owned_by: nvidiaId.split('/')[0],
      nvidia_model_id: nvidiaId,
      type: 'chat',
    });
  }

  for (const [alias, nvidiaId] of Object.entries(NVIDIA_MODELS.vision)) {
    localModels.push({
      id: alias,
      object: 'model',
      created: 1700000000,
      owned_by: nvidiaId.split('/')[0],
      nvidia_model_id: nvidiaId,
      type: 'vision',
    });
  }

  for (const [alias, nvidiaId] of Object.entries(NVIDIA_MODELS.embedding)) {
    localModels.push({
      id: alias,
      object: 'model',
      created: 1700000000,
      owned_by: nvidiaId.split('/')[0],
      nvidia_model_id: nvidiaId,
      type: 'embedding',
    });
  }

  for (const [alias, nvidiaId] of Object.entries(NVIDIA_MODELS.image)) {
    localModels.push({
      id: alias,
      object: 'model',
      created: 1700000000,
      owned_by: nvidiaId.split('/')[0],
      nvidia_model_id: nvidiaId,
      type: 'image',
    });
  }

  // 合并实时模型（如果有）
  const allModels = liveModels.length > 0
    ? [...localModels, ...liveModels.filter((m: any) =>
        !localModels.some(l => l.nvidia_model_id === m.id))]
    : localModels;

  return c.json({
    object: 'list',
    data: allModels,
    total: allModels.length,
  });
});

// 聊天补全端点
app.post('/v1/chat/completions', async (c) => {
  const apiKey = c.env.NVIDIA_API_KEY;

  if (!apiKey) {
    return c.json({
      error: {
        message: 'NVIDIA API Key not configured. Please set NVIDIA_API_KEY environment variable.',
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

    // 获取 NVIDIA 模型 ID
    const nvidiaModel = getNvidiaModelId(body.model);

    // 处理消息内容
    const processedMessages = body.messages.map(msg => {
      if (typeof msg.content === 'string') {
        return msg;
      }
      // 处理多模态内容
      const textParts: string[] = [];
      const imageUrls: string[] = [];

      for (const part of msg.content) {
        if (part.type === 'text' && part.text) {
          textParts.push(part.text);
        } else if (part.type === 'image_url' && part.image_url) {
          imageUrls.push(part.image_url.url);
        }
      }

      return {
        role: msg.role,
        content: textParts.join('\n'),
        ...(imageUrls.length > 0 && { images: imageUrls })
      };
    });

    // 构建 NVIDIA NIM API 请求
    const nvidiaRequest = {
      model: nvidiaModel,
      messages: processedMessages,
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

    const data = await response.json() as {
      id?: string;
      object?: string;
      created?: number;
      choices?: Array<{
        index: number;
        message?: { role?: string; content?: string };
        finish_reason?: string;
      }>;
      usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    };

    // 转换响应格式以保持 OpenAI 兼容性
    return c.json({
      id: data.id,
      object: data.object,
      created: data.created,
      model: body.model,
      choices: data.choices?.map((choice) => ({
        index: choice.index,
        message: {
          role: choice.message?.role || 'assistant',
          content: choice.message?.content || '',
        },
        finish_reason: choice.finish_reason,
      })) || [],
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

// 嵌入端点
app.post('/v1/embeddings', async (c) => {
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
    const body = await c.req.json<EmbeddingRequest>();

    if (!body.input) {
      return c.json({
        error: {
          message: 'input is required',
          type: 'invalid_request_error',
          code: 'invalid_input'
        }
      }, 400);
    }

    const nvidiaModel = getNvidiaModelId(body.model);

    // NVIDIA 嵌入 API 请求
    const response = await fetch('https://integrate.api.nvidia.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: nvidiaModel,
        input: body.input,
        encoding_format: body.encoding_format || 'float',
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return c.json({
        error: {
          message: `NVIDIA API error: ${response.status}`,
          details: errorText,
        }
      }, response.status as 400 | 401 | 403 | 404 | 429 | 500 | 502 | 503);
    }

    const data = await response.json() as {
      data?: Array<{ object: string; index: number; embedding: number[] }>;
      embeddings?: Array<{ embedding?: number[] } | number[]>;
      usage?: { prompt_tokens: number; total_tokens: number };
    };

    return c.json({
      object: 'list',
      data: data.data || data.embeddings?.map((emb, index) => ({
        object: 'embedding',
        index,
        embedding: 'embedding' in emb ? emb.embedding : emb,
      })) || [],
      model: body.model,
      usage: data.usage || { prompt_tokens: 0, total_tokens: 0 },
    });

  } catch (error) {
    console.error('Embedding error:', error);
    return c.json({
      error: {
        message: error instanceof Error ? error.message : 'Internal server error',
        type: 'internal_error',
      }
    }, 500);
  }
});

// 图像生成端点
app.post('/v1/images/generations', async (c) => {
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
    const body = await c.req.json<ImageGenerationRequest>();

    if (!body.prompt) {
      return c.json({
        error: {
          message: 'prompt is required',
          type: 'invalid_request_error',
          code: 'invalid_prompt'
        }
      }, 400);
    }

    const nvidiaModel = getNvidiaModelId(body.model || 'sdxl');

    // NVIDIA 图像生成 API 请求
    const response = await fetch('https://integrate.api.nvidia.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: nvidiaModel,
        prompt: body.prompt,
        n: body.n || 1,
        size: body.size || '1024x1024',
        quality: body.quality,
        style: body.style,
        response_format: body.response_format || 'url',
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return c.json({
        error: {
          message: `NVIDIA API error: ${response.status}`,
          details: errorText,
        }
      }, response.status as 400 | 401 | 403 | 404 | 429 | 500 | 502 | 503);
    }

    const data = await response.json() as {
      data?: Array<{ url?: string; b64_json?: string }>;
      images?: Array<{ url?: string; b64_json?: string } | string>;
    };

    return c.json({
      created: Date.now(),
      data: data.data || data.images?.map((img) => ({
        url: typeof img === 'string' ? img : img.url,
        b64_json: typeof img === 'object' ? img.b64_json : undefined,
      })) || [],
    });

  } catch (error) {
    console.error('Image generation error:', error);
    return c.json({
      error: {
        message: error instanceof Error ? error.message : 'Internal server error',
        type: 'internal_error',
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
