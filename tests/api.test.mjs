import test from 'node:test';
import assert from 'node:assert/strict';
import { embedBatchesAdaptive, normalizeBaseUrl, requestEmbeddings, requestModelList, requestRerank } from '../src/api.js';

function response(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload };
}

test('base URL normalization preserves provider version paths', () => {
  assert.equal(normalizeBaseUrl('https://example.test'), 'https://example.test/v1');
  assert.equal(normalizeBaseUrl('https://example.test/v1/'), 'https://example.test/v1');
  assert.equal(normalizeBaseUrl('https://generativelanguage.googleapis.com/v1beta/openai'), 'https://generativelanguage.googleapis.com/v1beta/openai');
});

test('embedding rows are sorted by index and dimensions are validated', async () => {
  let captured;
  const fetchMock = async (url, init) => {
    captured = { url, body: JSON.parse(init.body) };
    return response({ data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }] });
  };
  const vectors = await requestEmbeddings(['first', 'second'], {
    baseUrl: 'https://example.test/v1', apiKey: 'k', model: 'm', dimensions: 2
  }, { fetchImpl: fetchMock });
  assert.deepEqual(vectors, [[1, 0], [0, 1]]);
  assert.equal(captured.url, 'https://example.test/v1/embeddings');
  assert.deepEqual(captured.body.input, ['first', 'second']);
  assert.equal(captured.body.dimensions, 2);
});

test('batch limit errors halve batch size and retry', async () => {
  const calls = [];
  const fetchMock = async (_url, init) => {
    const body = JSON.parse(init.body);
    calls.push(body.input.length);
    if (calls.length === 1) return response({ error: { message: 'batch limit exceeded' } }, 400);
    return response({ data: body.input.map((_, index) => ({ index, embedding: [1, 0] })) });
  };
  const vectors = await embedBatchesAdaptive(['a', 'b', 'c'], {
    baseUrl: 'https://example.test', apiKey: 'k', model: 'm', batchSize: 2
  }, { fetchImpl: fetchMock });
  assert.equal(vectors.length, 3);
  assert.deepEqual(calls, [2, 1, 1, 1]);
});

test('jina rerank request parses relevance scores', async () => {
  let request;
  const rows = await requestRerank('query', ['a', 'b'], {
    rerankApiFormat: 'jina', rerankUrl: 'https://rerank.test/v1/rerank', rerankKey: 'k', rerankModel: 'm', topN: 2
  }, {
    fetchImpl: async (_url, init) => {
      request = JSON.parse(init.body);
      return response({ results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.2 }] });
    }
  });
  assert.deepEqual(request.documents, ['a', 'b']);
  assert.equal(rows[0].index, 1);
  assert.equal(rows[0].relevance_score, 0.9);
});

test('rerank HTTP failure is surfaced for runtime fallback', async () => {
  await assert.rejects(async () => {
    try {
      await requestRerank('query', ['a'], { rerankApiFormat: 'jina', rerankUrl: 'https://rerank.test' }, {
        fetchImpl: async () => response({ error: { message: 'server down' } }, 500)
      });
    } catch (error) {
      assert.equal(error.status, 500);
      throw error;
    }
  }, /server down/);
});

test('model list is fetched from /models, deduped and sorted', async () => {
  let captured;
  const models = await requestModelList({ baseUrl: 'https://example.test', apiKey: 'k' }, {
    fetchImpl: async (url, init) => {
      captured = { url, method: init.method, auth: init.headers.Authorization };
      return response({ data: [{ id: 'z-embed' }, { id: 'a-embed' }, { id: 'z-embed' }, { name: 'named-model' }] });
    }
  });
  assert.equal(captured.url, 'https://example.test/v1/models');
  assert.equal(captured.method, 'GET');
  assert.equal(captured.auth, 'Bearer k');
  assert.deepEqual(models, ['a-embed', 'named-model', 'z-embed']);
});
