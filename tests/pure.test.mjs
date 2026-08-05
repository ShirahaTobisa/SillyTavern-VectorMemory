import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyRerankResults,
  assembleTurnFragments,
  buildConversationTurns,
  cosineSimilarity,
  dedupeByFingerprint,
  getContentFingerprint,
  getTurnFingerprint,
  lexicalBoost,
  mergeSameTurnResults,
  quantizeEmbedding,
  decodeQuantizedEmbedding,
  rankVectorCandidates,
  splitLongMemoryParagraph,
  stripVectorMemoryCode
} from '../src/pure.js';

test('cleaning keeps prose and strips code/html', () => {
  const input = [
    '这是正常文本，应该保留。',
    '<script>console.log("secret")</script>',
    '```javascript\nconst secret = true;\n```',
    '<div>HTML only</div>',
    '正文继续。'
  ].join('\n');
  const output = stripVectorMemoryCode(input);
  assert.match(output, /这是正常文本/);
  assert.match(output, /正文继续/);
  assert.doesNotMatch(output, /console\.log|secret|HTML only|const secret/);
});

test('split boundaries honor 1800 and sentence-window behavior', () => {
  assert.equal(splitLongMemoryParagraph('a'.repeat(1800)).length, 1);
  const parts = splitLongMemoryParagraph('a'.repeat(1801));
  assert.equal(parts.length, 2);
  assert.ok(parts.every(part => part.length <= 1800));
  const sentence = `${'a'.repeat(1200)}。${'b'.repeat(900)}`;
  const sentenceParts = splitLongMemoryParagraph(sentence);
  assert.equal(sentenceParts[0].at(-1), '。');
  assert.ok(sentenceParts[0].length > 990);
});

test('fragment assembly carries paired labels, turn fingerprint and structural ids', () => {
  const messages = [
    { is_user: true, mes: '用户提出了一个足够长的问题。'.repeat(12), name: 'U' },
    { is_user: false, role: 'assistant', mes: '角色回答第一段。'.repeat(30), name: 'A' },
    { is_user: false, role: 'assistant', mes: '角色回答第二段。'.repeat(30), name: 'A' }
  ];
  const [turn] = buildConversationTurns(messages);
  const fragments = assembleTurnFragments(turn, { userName: 'U', roleName: 'A' });
  assert.ok(fragments.length >= 2);
  assert.ok(fragments.every(fragment => fragment.paragraph.includes('用户：') && fragment.paragraph.includes('角色卡：')));
  assert.ok(fragments.every(fragment => fragment.sourceText.startsWith('第 1 轮\n')));
  assert.ok(fragments.every(fragment => fragment.vectorChunkId.startsWith('1:')));
  assert.ok(fragments.every(fragment => fragment.turnFingerprint.length >= 80));
});

test('quantization round trip preserves cosine closely', () => {
  const vector = [0.1, -0.2, 0.45, 1, -0.75, 0.03];
  const packed = quantizeEmbedding(vector);
  const decoded = decodeQuantizedEmbedding(packed.embeddingQ);
  const error = Math.abs(cosineSimilarity(vector, decoded) - 1);
  assert.ok(error < 0.01, `cosine error ${error}`);
  assert.equal(packed.embeddingEncoding, 'int8:maxabs:v1');
  assert.equal(packed.embeddingDims, vector.length);
});

test('content fingerprint and dedupe are idempotent', () => {
  const paragraph = '同一段内容用于验证去重行为。'.repeat(12);
  assert.ok(getContentFingerprint(paragraph));
  const items = [{ turn: 1, sequence: 1, paragraph }, { turn: 1, sequence: 1, paragraph }];
  const once = dedupeByFingerprint(items);
  const twice = dedupeByFingerprint([...once, ...once]);
  assert.equal(once.length, 1);
  assert.equal(twice.length, 1);
});

test('lexical boost and score ordering follow RPH rules', () => {
  const terms = ['记忆', 'qwen'];
  const lexical = lexicalBoost({ sourceText: '记忆 qwen', summary: '' }, terms);
  assert.equal(lexical.hits, 2);
  assert.equal(lexical.boost, 0.03);
  const query = [1, 0];
  const ranked = rankVectorCandidates([
    { turn: 1, sequence: 1, paragraph: 'a'.repeat(100), embedding: [0.8, 0.2], enabled: true },
    { turn: 2, sequence: 1, paragraph: 'b'.repeat(100), embedding: [0.8, 0.2], enabled: true }
  ], query, '', { threshold: 50 });
  assert.equal(ranked[0].memory.turn, 2, 'ties prefer newer turns');
});

test('same-turn hits merge to current full turn text', () => {
  const messages = [
    { is_user: true, mes: '用户事实'.repeat(30) },
    { is_user: false, role: 'assistant', mes: '回答一'.repeat(30) },
    { is_user: false, role: 'assistant', mes: '回答二'.repeat(30) }
  ];
  const [turn] = buildConversationTurns(messages);
  const fingerprint = getTurnFingerprint(turn);
  const items = [1, 2].map((sequence, index) => ({
    memory: { turn: 1, sequence, turnFingerprint: fingerprint, paragraph: `用户：x\n角色卡：${index}` },
    rawScore: 0.7 + index * 0.01,
    score: 0.7 + index * 0.01
  }));
  const merged = mergeSameTurnResults(items, [turn]);
  assert.equal(merged.length, 1);
  assert.match(merged[0].memory.paragraph, /回答一/);
  assert.match(merged[0].memory.paragraph, /回答二/);
});

test('rerank threshold selection and explicit fallback are testable without ST', () => {
  const items = [{ memory: { turn: 1 }, score: 0.8 }, { memory: { turn: 2 }, score: 0.7 }];
  const selected = applyRerankResults(items, [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.2 }], { topK: 2, threshold: 0.35 });
  assert.equal(selected.usedFallback, false);
  assert.equal(selected.items[0].memory.turn, 2);
  const fallback = applyRerankResults(items, null, { topK: 1, threshold: 0.35 });
  assert.equal(fallback.usedFallback, true);
  assert.equal(fallback.items[0].memory.turn, 1);
});
