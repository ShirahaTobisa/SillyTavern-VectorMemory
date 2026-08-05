/* Network adapters. All requests accept an injected fetch implementation for tests. */

export const DEFAULT_TIMEOUT_MS = 60_000;

export function normalizeBaseUrl(value) {
  const raw = String(value || '').trim().replace(/\/+$/, '');
  if (!raw) return '';
  // Gemini's OpenAI-compatible path already contains /v1beta/openai. Keeping
  // it intact is required for the preset; ordinary providers get /v1.
  if (/\/v1(?:beta)?(?:\/|$)/i.test(raw) || /\/openai$/i.test(raw)) return raw;
  return `${raw}/v1`;
}

export function getEmbeddingUrl(baseUrl) {
  const normalized = normalizeBaseUrl(baseUrl);
  return normalized ? `${normalized}/embeddings` : '';
}

function getFetch(fetchImpl) {
  const implementation = fetchImpl || globalThis.fetch;
  if (typeof implementation !== 'function') throw new Error('当前环境没有可用的 fetch');
  return implementation;
}

function abortError() {
  const error = new Error('Aborted');
  error.name = 'AbortError';
  return error;
}

async function fetchWithTimeout(url, init, options = {}) {
  const timeoutMs = Number(options.timeoutMs) || DEFAULT_TIMEOUT_MS;
  const externalSignal = options.signal;
  if (externalSignal?.aborted) throw abortError();
  const controller = new AbortController();
  let timer;
  const forwardAbort = () => controller.abort();
  externalSignal?.addEventListener?.('abort', forwardAbort, { once: true });
  timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await getFetch(options.fetchImpl)(url, { ...init, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) {
      if (externalSignal?.aborted) throw abortError();
      const timeoutError = new Error(`请求超时（${timeoutMs}ms）`);
      timeoutError.name = 'TimeoutError';
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener?.('abort', forwardAbort);
  }
}

export function extractApiError(payload, status) {
  if (payload == null) return `API Error: ${status}`;
  if (typeof payload === 'string') return payload;
  return payload.error?.message
    || payload.message
    || payload.msg
    || payload.code && `${payload.code}: ${payload.message || ''}`
    || `API Error: ${status}`;
}

function normalizeEmbedding(value) {
  const raw = Array.isArray(value) || ArrayBuffer.isView(value)
    ? value
    : (Array.isArray(value?.values) || ArrayBuffer.isView(value?.values) ? value.values : []);
  return Array.from(raw, Number).filter(Number.isFinite);
}

export async function requestEmbeddings(inputs, config = {}, options = {}) {
  const values = (Array.isArray(inputs) ? inputs : [inputs]).map(input => String(input || '').trim());
  if (values.length === 0 || values.some(input => !input)) throw new Error('嵌入内容不能为空');
  const model = String(config.model || '').trim();
  const baseUrl = normalizeBaseUrl(config.baseUrl);
  if (!baseUrl || !model) throw new Error('请先配置 embedding API 地址和模型');

  const body = { model, input: values };
  const dimensions = Number(config.dimensions);
  if (Number.isFinite(dimensions) && dimensions > 0) body.dimensions = Math.round(dimensions);
  const headers = { 'Content-Type': 'application/json' };
  if (config.apiKey !== undefined) headers.Authorization = `Bearer ${String(config.apiKey || '')}`;
  const response = await fetchWithTimeout(`${baseUrl}/embeddings`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body)
  }, options);

  let payload;
  try { payload = await response.json(); } catch (_) { payload = null; }
  if (!response.ok) {
    const error = new Error(extractApiError(payload, response.status));
    error.status = response.status;
    throw error;
  }
  const rows = Array.isArray(payload?.data) ? [...payload.data] : [];
  rows.sort((a, b) => (Number(a?.index) || 0) - (Number(b?.index) || 0));
  const vectors = rows.map(row => normalizeEmbedding(row?.embedding));
  if (vectors.length !== values.length || vectors.some(vector => vector.length === 0)) {
    throw new Error('embedding 接口返回的数据数量或维度不完整');
  }
  const dims = vectors[0].length;
  if (vectors.some(vector => vector.length !== dims)) throw new Error('embedding 接口返回的向量维度不一致');
  if (Number.isFinite(dimensions) && dimensions > 0 && dims !== Math.round(dimensions)) {
    throw new Error(`embedding 维度不匹配：期望 ${Math.round(dimensions)}，实际 ${dims}`);
  }
  return vectors;
}

export async function embedBatchesAdaptive(items, config = {}, options = {}) {
  const values = (Array.isArray(items) ? items : []).map(item => String(item || '').trim()).filter(Boolean);
  if (values.length === 0) return [];
  let batchSize = Math.max(1, Math.min(16, Math.round(Number(config.batchSize) || 8)));
  const vectors = [];
  let index = 0;
  let consecutiveFailures = 0;

  while (index < values.length) {
    if (options.signal?.aborted) throw abortError();
    const batch = values.slice(index, index + batchSize);
    try {
      const result = await requestEmbeddings(batch, config, options);
      vectors.push(...result);
      index += batch.length;
      consecutiveFailures = 0;
      options.onProgress?.(index, values.length);
    } catch (error) {
      if (error?.name === 'AbortError') throw error;
      consecutiveFailures += 1;
      if (consecutiveFailures >= 3) throw error;
      if (responseLooksLikeBatchLimit(error) && batchSize > 1) {
        batchSize = Math.max(1, Math.floor(batchSize / 2));
        continue;
      }
      // Give transient provider errors two retries. The short delay is
      // intentionally interruptible so CHAT_CHANGED can stop a patrol.
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, 250);
        if (options.signal) {
          const stop = () => { clearTimeout(timer); reject(abortError()); };
          options.signal.addEventListener('abort', stop, { once: true });
        }
      });
    }
  }
  return vectors;
}

function responseLooksLikeBatchLimit(error) {
  const message = String(error?.message || error || '').toLowerCase();
  return /batch|limit|exceed|too many|max(imum)?|数量|上限|超出/.test(message);
}

export async function requestModelList(config = {}, options = {}) {
  const baseUrl = normalizeBaseUrl(config.baseUrl);
  if (!baseUrl) throw new Error('请先配置 API 地址');
  const headers = {};
  if (config.apiKey !== undefined) headers.Authorization = `Bearer ${String(config.apiKey || '')}`;
  const response = await fetchWithTimeout(`${baseUrl}/models`, { method: 'GET', headers }, options);
  let payload;
  try { payload = await response.json(); } catch (_) { payload = null; }
  if (!response.ok) {
    const error = new Error(extractApiError(payload, response.status));
    error.status = response.status;
    throw error;
  }
  const rows = Array.isArray(payload?.data) ? payload.data : (Array.isArray(payload?.models) ? payload.models : []);
  const ids = rows.map(row => String(row?.id || row?.name || '')).filter(Boolean);
  return [...new Set(ids)].sort((a, b) => a.localeCompare(b));
}

export async function requestRerank(query, documents, config = {}, options = {}) {
  const format = config.rerankApiFormat === 'dashscope' ? 'dashscope' : 'jina';
  const values = (Array.isArray(documents) ? documents : []).map(item => String(item || ''));
  if (!String(query || '').trim() || values.length === 0) return [];
  const headers = { 'Content-Type': 'application/json' };
  if (config.rerankKey !== undefined) headers.Authorization = `Bearer ${String(config.rerankKey || '')}`;

  let url;
  let body;
  if (format === 'dashscope') {
    url = String(config.rerankUrl || 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank');
    body = {
      model: String(config.rerankModel || 'Qwen/Qwen3-Reranker-8B'),
      input: { query: String(query), documents: values },
      parameters: { top_n: Number(config.topN) || values.length, return_documents: false }
    };
  } else {
    url = String(config.rerankUrl || '').trim();
    if (!url) throw new Error('未配置 rerank URL');
    body = {
      model: String(config.rerankModel || 'Qwen/Qwen3-Reranker-8B'),
      query: String(query),
      documents: values,
      top_n: Number(config.topN) || values.length
    };
  }

  const response = await fetchWithTimeout(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body)
  }, options);
  let payload;
  try { payload = await response.json(); } catch (_) { payload = null; }
  if (!response.ok) {
    const error = new Error(extractApiError(payload, response.status));
    error.status = response.status;
    throw error;
  }
  const rows = format === 'dashscope' ? payload?.output?.results : payload?.results;
  if (!Array.isArray(rows)) throw new Error('rerank 接口返回的数据不完整');
  return rows.map(row => ({
    index: Number(row?.index),
    relevance_score: Number(row?.relevance_score)
  })).filter(row => Number.isInteger(row.index) && Number.isFinite(row.relevance_score));
}
