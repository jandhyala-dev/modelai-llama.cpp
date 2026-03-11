# ModelAI Two-Process Architecture — Replacing Ollama

## Goal

Eliminate the Ollama dependency entirely. ModelAI currently uses Ollama for:

1. **Chat completions** — `/api/chat` (Qwen 2.5 14B)
2. **Embeddings** — `/api/embeddings` (BGE-M3, 1024 dimensions)
3. **Reranking** — via chat-based scoring prompts (same chat model)
4. **Model listing** — `/api/tags`
5. **Health checks** — `/api/tags`

modelai-llama.cpp replaces all five with two supervised processes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelAI Node.js Backend                   │
│                                                             │
│  ┌─────────────────────┐   ┌──────────────────────────────┐ │
│  │  llama-supervisor    │   │  runtime-registry.cjs        │ │
│  │                      │   │                              │ │
│  │  Spawns & manages:   │   │  Routes requests:            │ │
│  │  • chat process      │   │  • chat → port 8090          │ │
│  │  • embed process     │   │  • embed → port 8091         │ │
│  │                      │   │  • rerank → port 8090 (chat) │ │
│  └──────────┬───────────┘   └──────────────┬───────────────┘ │
│             │                              │                 │
└─────────────┼──────────────────────────────┼─────────────────┘
              │                              │
    ┌─────────▼──────────┐        ┌──────────▼──────────┐
    │  PROCESS 1: CHAT   │        │  PROCESS 2: EMBED   │
    │  Port 8090         │        │  Port 8091          │
    │                    │        │                     │
    │  Model: Qwen 2.5   │        │  Model: BGE-M3      │
    │  14B Q4_K_M        │        │  F16 (1.1GB)        │
    │                    │        │                     │
    │  Endpoints:        │        │  Endpoints:         │
    │  /health           │        │  /health            │
    │  /props            │        │  /v1/embeddings     │
    │  /v1/models        │        │  /v1/models         │
    │  /v1/chat/complete │        │  /props             │
    │  /metrics          │        │                     │
    │  /slots            │        │                     │
    └────────────────────┘        └─────────────────────┘
```

## Process Details

### Process 1: Chat (port 8090)

```bash
llama-server \
  --host 127.0.0.1 \
  --port 8090 \
  --model /path/to/qwen2.5-14b-instruct-q4_k_m.gguf \
  --ctx-size 16384 \
  --parallel 2 \
  --metrics \
  --slots \
  --jinja \
  --flash-attn
```

Handles: chat completions, reranking (via generation), tool use, streaming, structured output.

### Process 2: Embeddings (port 8091)

```bash
llama-server \
  --host 127.0.0.1 \
  --port 8091 \
  --model /path/to/bge-m3-f16.gguf \
  --ctx-size 512 \
  --parallel 4 \
  --embedding \
  --pooling mean
```

Handles: embedding generation only. 4 parallel slots because embedding requests are fast and bursty (batch re-embed workers send 5 at a time).

Memory footprint: ~1.5GB (model 1.1GB + context). Negligible next to the 14B chat model.

## Supervisor Changes (llama-supervisor.cjs)

### Current: Single process

```javascript
// Current
const proc = spawn(binaryPath, chatArgs);
await pollHealth(`http://127.0.0.1:8090/health`);
```

### New: Two managed processes

```javascript
class LlamaSupervisor {
  constructor(config) {
    this.processes = {
      chat:  { port: 8090, proc: null, ready: false },
      embed: { port: 8091, proc: null, ready: false },
    };
  }

  async start() {
    // Spawn both in parallel
    this.processes.chat.proc  = spawn(binaryPath, chatArgs);
    this.processes.embed.proc = spawn(binaryPath, embedArgs);

    // Wait for both to be ready (parallel health polls)
    await Promise.all([
      this.pollHealth('chat',  8090),
      this.pollHealth('embed', 8091),
    ]);

    // Validate contracts
    await this.validateProps('chat',  8090);
    await this.validateProps('embed', 8091);
  }

  async stop() {
    // SIGINT both, wait, SIGKILL if needed
    for (const key of ['chat', 'embed']) {
      const p = this.processes[key];
      if (p.proc) {
        p.proc.kill('SIGINT');
        await this.waitForExit(p.proc, 5000);
      }
    }
  }

  async restart(which) {
    // Restart individual process (up to 3 attempts)
    // 'chat' or 'embed'
  }
}
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODELAI_LLAMA_CPP_BIN` | (required) | Path to llama-server binary |
| `MODELAI_LLAMA_CPP_MODEL` | (required) | Path to chat model GGUF |
| `MODELAI_LLAMA_CPP_EMBED_MODEL` | (optional) | Path to embedding model GGUF. If unset, embeddings fall back to Ollama. |
| `MODELAI_LLAMA_CPP_CHAT_PORT` | `8090` | Chat server port |
| `MODELAI_LLAMA_CPP_EMBED_PORT` | `8091` | Embedding server port |
| `MODELAI_LLAMA_CPP_CHAT_CTX` | `16384` | Chat context window |
| `MODELAI_LLAMA_CPP_EMBED_CTX` | `512` | Embedding context window |
| `MODELAI_LLAMA_CPP_CHAT_PARALLEL` | `2` | Chat parallel slots |
| `MODELAI_LLAMA_CPP_EMBED_PARALLEL` | `4` | Embedding parallel slots |
| `MODELAI_LLAMA_CPP_MAX_RESTARTS` | `3` | Max restart attempts per process |

## Adapter Changes (llamacpp-adapter.cjs)

### Request Routing

```javascript
class LlamaCppAdapter {
  constructor(chatPort = 8090, embedPort = 8091) {
    this.chatBase  = `http://127.0.0.1:${chatPort}`;
    this.embedBase = `http://127.0.0.1:${embedPort}`;
  }

  // Chat, reranking, tool use → chat process
  async infer(params)  { return fetch(`${this.chatBase}/v1/chat/completions`, ...); }
  async stream(params) { return fetch(`${this.chatBase}/v1/chat/completions`, ...); }
  async health()       { return fetch(`${this.chatBase}/health`); }
  async getProps()     { return fetch(`${this.chatBase}/props`); }
  async getModels()    { return fetch(`${this.chatBase}/v1/models`); }

  // Embeddings → embed process
  async embed(text) {
    const res = await fetch(`${this.embedBase}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: text, model: 'bge-m3' }),
    });
    const data = await res.json();
    return {
      embedding: data.data[0].embedding,  // float[1024]
      model: data.model,
      dimensions: data.data[0].embedding.length,
    };
  }

  async embedBatch(texts) {
    // llama-server /v1/embeddings supports array input
    const res = await fetch(`${this.embedBase}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: texts, model: 'bge-m3' }),
    });
    const data = await res.json();
    return data.data.map(d => ({
      embedding: d.embedding,
      dimensions: d.embedding.length,
    }));
  }
}
```

### Embedding Module Changes (embeddings.cjs)

Replace `embedWithOllama()` calls:

```javascript
// Before
async function embed(text) {
  const model = await detectOllamaModel();  // bge-m3 > nomic > mxbai > all-minilm
  return embedWithOllama(text, model);       // POST /api/embeddings
}

// After
async function embed(text) {
  const adapter = getLocalAdapter();  // from runtime-registry
  if (adapter.type === 'modelai-llama.cpp') {
    return adapter.embed(text);       // POST /v1/embeddings on port 8091
  }
  // Fallback: Ollama or OpenAI
  const model = await detectOllamaModel();
  return embedWithOllama(text, model);
}
```

### Reembed Worker Changes (reembed-worker.cjs)

The reembed worker currently calls `embeddings.embed()` which calls Ollama. Since `embed()` now routes through the adapter, the worker needs no changes — it automatically uses the embed process.

For batch mode, the worker can use `adapter.embedBatch()` directly for better throughput (single HTTP call for batch of 5).

## Health & Readiness

### Composite Health Check

The supervisor reports both processes:

```javascript
async compositeHealth() {
  const [chat, embed] = await Promise.allSettled([
    fetch(`${this.chatBase}/health`),
    fetch(`${this.embedBase}/health`),
  ]);
  return {
    chat:  chat.status === 'fulfilled' && chat.value.ok,
    embed: embed.status === 'fulfilled' && embed.value.ok,
    ready: /* both true */,
  };
}
```

ModelAI marks the runtime as ready only when both processes pass health checks.

### Graceful Degradation

If the embed process dies but chat is healthy:
- Chat, reranking, tool use continue working
- Embedding requests fail with a clear error
- Supervisor attempts restart (up to MAX_RESTARTS)
- If restart fails, ModelAI can fall back to Ollama for embeddings only

## Memory Budget (Apple Silicon)

| Component | RAM |
|-----------|-----|
| Chat model (Qwen 2.5 14B Q4_K_M) | ~9 GB |
| Chat KV cache (16K ctx, 2 slots) | ~1 GB |
| Embed model (BGE-M3 F16) | ~1.1 GB |
| Embed KV cache (512 ctx, 4 slots) | ~0.05 GB |
| **Total** | **~11.2 GB** |

Fits comfortably on 16GB M-series. For 8GB machines, use a smaller chat model (7B) or reduce context/slots.

## Migration Path to PG Vector (Supabase)

When moving to cloud on Supabase with pgvector:

1. **Indexing phase** — embeddings computed locally by the embed process, stored in Supabase pgvector via `INSERT INTO ... embedding = $1`
2. **Query phase** — only the query embedding is needed (single call to embed process)
3. **Similarity search** — `ORDER BY embedding <=> $query_embedding LIMIT k` runs server-side in Postgres
4. **Transition** — Qdrant is replaced by pgvector. The embed process stays the same. Only `vector-store.cjs` changes (Qdrant client → Supabase/pg client).

The embed process remains useful even with pgvector — you still need to compute embeddings for new documents and queries.

## Validated Configuration

Tested and confirmed working on this fork:

```
Binary:      /Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/build-release/bin/llama-server
Chat model:  /Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/models/test/qwen2.5-7b-instruct-q3_k_m.gguf
             /Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/models/test/qwen2.5-14b-instruct-q2_k-00001-of-00002.gguf
             /Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/models/test/deepseek-r1-distill-qwen-14b-q3_k_m.gguf
Embed model: /Users/ajayjandhyala/dev/whippet/modelai-llama.cpp/models/test/bge-m3-f16.gguf

Contract:    modelai.contract.contract_version = "0.1.0"
Embed dims:  1024 (matches BGE-M3 spec and Qdrant vectors_bq collection)
Embed API:   /v1/embeddings (OpenAI-compatible)
```
