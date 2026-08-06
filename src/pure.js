/* Pure vector-memory algorithms. This module intentionally has no SillyTavern dependencies. */

export const MEMORY_MAX_PARAGRAPH_LENGTH = 1800;
export const MEMORY_MERGE_MAX_LENGTH = 400;
export const MEMORY_MIN_TOP_K = 10;
export const MEMORY_MAX_TOP_K = 20;
export const MEMORY_DEFAULT_TOP_K = 10;
export const MEMORY_DEFAULT_SIMILARITY_THRESHOLD = 50;

const CONTENT_PUNCTUATION = /[，。、“”‘’：；！？,.!?;:"'`~]/g;

/**
 * Card-specific wrapper blocks (planning CoT, state trackers, selection menus…)
 * whose INNER TEXT is junk for embeddings. Entries support `*` wildcards and
 * match paired tags wholesale; users extend the list per card in settings.
 */
export const DEFAULT_STRIP_TAGS = 'think*, thinking, thought*, cot*, plan, *planning*, analysis, updatevariable, jsonpatch, current_event, progress, selection, summary, status*, konatan_*';

export function compileStripTagList(list) {
  const entries = String(list || '').split(/[,，\n]+/).map(entry => entry.trim()).filter(Boolean);
  return entries.map(entry => {
    const namePattern = entry
      .replace(/[.*+?^${}()|[\]\\]/g, ch => (ch === '*' ? '\u0000' : `\\${ch}`))
      .replace(/\u0000/g, '[^<>/\\s]*');
    return {
      pair: new RegExp(`<\\s*(${namePattern})(?:\\s[^>]*)?>[\\s\\S]*?</\\s*\\1\\s*>`, 'gi'),
      orphanClose: new RegExp(`^[\\s\\S]*?</\\s*(${namePattern})\\s*>`, 'i'),
      orphanOpen: new RegExp(`<\\s*(${namePattern})(?:\\s[^>]*)?>[\\s\\S]*$`, 'i')
    };
  });
}

let activeTagBlockRegexes = compileStripTagList(DEFAULT_STRIP_TAGS);

// Paired blocks go first; then prefill-opened blocks (only a closing tag in
// the message) and truncated blocks (an opener with no closer).
function stripConfiguredTagBlocks(text) {
  let result = text;
  activeTagBlockRegexes.forEach(entry => { result = result.replace(entry.pair, ''); });
  activeTagBlockRegexes.forEach(entry => {
    result = result.replace(entry.orphanClose, '');
    result = result.replace(entry.orphanOpen, '');
  });
  return result;
}

export function setStripTagList(list) {
  const value = list == null || String(list).trim() === '' ? DEFAULT_STRIP_TAGS : String(list);
  activeTagBlockRegexes = compileStripTagList(value);
}

export function trimText(text, maxLength) {
  const cleanText = String(text || '').replace(/\n{3,}/g, '\n\n').trim();
  if (!Number.isFinite(maxLength) || cleanText.length <= maxLength) return cleanText;
  return `${cleanText.slice(0, maxLength)}...`;
}

/**
 * Keep this order in sync with the RPH implementation. It is deliberately
 * conservative about ordinary prose and aggressive about executable markup.
 */
export function stripVectorMemoryCode(text) {
  if (!text) return '';

  let result = String(text);
  result = stripConfiguredTagBlocks(result);
  result = result
    .replace(/image###[\s\S]*?###/gi, '')
    .replace(/<think(?:ing)?\b[^>]*>[\s\S]*?<\/think(?:ing)?>/gi, '')
    .replace(/<image>[\s\S]*?<\/image>/gi, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/~~~[\s\S]*?~~~/g, '')
    .replace(/<!DOCTYPE[\s\S]*?>/gi, '')
    .replace(/<html[\s\S]*?<\/html>/gi, '')
    .replace(/<(script|style|template|svg|canvas|iframe|object|embed|head|link|meta)[\s\S]*?<\/\1>/gi, '')
    .replace(/<(script|style|template|svg|canvas|iframe|object|embed|link|meta|input|img|br|hr)\b[^>]*\/?\s*>/gi, '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/`[^`\n]{1,200}`/g, '');

  const lines = result.split(/\r?\n/);
  const cleanedLines = [];

  const isCodeLikeLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) return false;
    if (/^<\/?[^\s<>]{1,30}\/?>$/.test(trimmed)) return true;
    if (/^<\/?[a-z][\w:-]*(\s|>|\/>)/i.test(trimmed)) return true;
    if (/^[{}()[\];,]+$/.test(trimmed)) return true;
    if (/^(const|let|var|function|class|import|export|return|if|else|for|while|switch|try|catch)\b/.test(trimmed)) return true;
    if (/^(#include|using\s+namespace|public:|private:|protected:|def\s+|from\s+\S+\s+import\s+)/.test(trimmed)) return true;
    if (/^(@click|v-if|v-for|v-model|class=|style=|id=|data-|aria-)/i.test(trimmed)) return true;
    if (/^[.#]?[a-zA-Z0-9_-]+\s*\{/.test(trimmed)) return true;
    if (/[{};]/.test(trimmed) && /(=>|===|!==|&&|\|\||;\s*$|:\s*function|\bconsole\.|\bdocument\.|\bwindow\.)/.test(trimmed)) return true;
    if (/<\/?[a-z][\w:-]*[\s\S]*?>/i.test(trimmed) && !/[，。！？、]/.test(trimmed)) return true;
    return false;
  };

  lines.forEach(line => {
    if (!isCodeLikeLine(line)) cleanedLines.push(line);
  });

  return cleanedLines.join('\n')
    .replace(/<\/?[a-z][\w:-]*\b[^>]*>/gi, '')
    .replace(/<\/?[^\s<>]{1,30}\/?>/g, '')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#039;/gi, "'")
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function cleanMessageText(message) {
  if (!message) return '';
  return stripVectorMemoryCode(message.mes ?? message.content ?? '').trim();
}

export function splitLongMemoryParagraph(paragraph, maxLength = MEMORY_MAX_PARAGRAPH_LENGTH) {
  const text = String(paragraph || '').trim();
  if (!text) return [];
  if (text.length <= maxLength) return [text];

  const parts = [];
  let remaining = text;
  while (remaining.length > maxLength) {
    const windowText = remaining.slice(0, maxLength);
    const breakAt = Math.max(
      windowText.lastIndexOf('。'),
      windowText.lastIndexOf('！'),
      windowText.lastIndexOf('？'),
      windowText.lastIndexOf('.'),
      windowText.lastIndexOf('!'),
      windowText.lastIndexOf('?'),
      windowText.lastIndexOf('\n')
    );
    const cutAt = breakAt > Math.floor(maxLength * 0.55) ? breakAt + 1 : maxLength;
    parts.push(remaining.slice(0, cutAt).trim());
    remaining = remaining.slice(cutAt).trim();
  }
  if (remaining) parts.push(remaining);
  return parts.filter(Boolean);
}

export function splitMemoryParagraphs(text, maxLength = MEMORY_MAX_PARAGRAPH_LENGTH) {
  const cleanText = String(text || '')
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  if (!cleanText) return [];
  return cleanText
    .split(/\n\s*\n/g)
    .map(p => p.trim())
    .filter(Boolean)
    .flatMap(paragraph => splitLongMemoryParagraph(paragraph, maxLength));
}

export function mergeSmallMemoryParagraphs(paragraphs, maxLength = MEMORY_MERGE_MAX_LENGTH) {
  const merged = [];
  let current = null;
  const flush = () => {
    if (current) merged.push(current);
    current = null;
  };

  (Array.isArray(paragraphs) ? paragraphs : []).forEach((paragraph, index) => {
    const text = String(paragraph || '').trim();
    if (!text) return;
    const paragraphNo = index + 1;
    if (!current) {
      current = { text, start: paragraphNo, end: paragraphNo };
      return;
    }
    const candidate = `${current.text}\n\n${text}`;
    if (candidate.length <= maxLength) {
      current.text = candidate;
      current.end = paragraphNo;
      return;
    }
    flush();
    current = { text, start: paragraphNo, end: paragraphNo };
  });
  flush();
  return merged;
}

export function buildConversationTurns(messages, { includeHidden = false } = {}) {
  const turns = [];
  let current = null;
  let sawUser = false;
  let turn = 0;

  (Array.isArray(messages) ? messages : []).forEach((message, index) => {
    if (!message || message.extra?.vector_memory_injected === true) return;
    // Hidden (is_system) floors are excluded from the prompt by ST itself, but
    // they are exactly what belongs in vector memory — extraction passes
    // includeHidden: true, prompt-side compression keeps the default.
    if (message.is_system === true && !includeHidden) return;
    const isUser = message.is_user === true || message.role === 'user';

    if (isUser) {
      if (current) turns.push(current);
      turn += 1;
      sawUser = true;
      current = { turn, messages: [], messageIndexes: [] };
      current.messages.push(message);
      current.messageIndexes.push(index);
      return;
    }

    // A first assistant message is the opening message and belongs to turn 0.
    if (!sawUser || !current) return;
    current.messages.push(message);
    current.messageIndexes.push(index);
  });
  if (current) turns.push(current);
  return turns;
}

export function getTurnFingerprint(turnInfo) {
  const parts = [];
  (turnInfo?.messages || []).forEach(message => {
    const text = cleanMessageText(message);
    if (!text) return;
    const label = message.is_user === true || message.role === 'user' ? '用户：' : '角色卡：';
    parts.push(`${label}${text}`);
  });
  const normalized = normalizeFingerprintText(parts.join('\n'));
  return normalized.length >= 80 ? normalized.slice(0, 1000) : '';
}

export function normalizeFingerprintText(text) {
  return String(text || '').replace(/\s+/g, '').replace(CONTENT_PUNCTUATION, '');
}

export function getContentFingerprint(text) {
  const normalized = normalizeFingerprintText(text);
  return normalized.length >= 80 ? normalized.slice(0, 1000) : '';
}

export function getRecallFingerprint(memory) {
  const text = String(memory?.paragraph || memory?.summary || memory?.sourceText || '').trim();
  const normalized = normalizeFingerprintText(text);
  if (normalized.length >= 80) return normalized.slice(0, 1000);
  return `${memory?.turn || ''}:${memory?.sequence || ''}:${normalized}`;
}

export function dedupeByFingerprint(items) {
  const seen = new Set();
  return (Array.isArray(items) ? items : []).filter(item => {
    const fingerprint = getRecallFingerprint(item);
    if (!fingerprint || seen.has(fingerprint)) return false;
    seen.add(fingerprint);
    return true;
  });
}

export function assembleTurnFragments(turnInfo, options = {}) {
  const userName = options.userName || '用户';
  const roleName = options.roleName || 'AI';
  const turn = Number(turnInfo?.turn) || 0;
  const userBlocks = [];
  const roleBlocks = [];

  (turnInfo?.messages || []).forEach((message, messagePosition) => {
    const isUser = message.is_user === true || message.role === 'user';
    const role = isUser ? 'user' : 'assistant';
    const groups = mergeSmallMemoryParagraphs(
      splitMemoryParagraphs(cleanMessageText(message)).flatMap(paragraph => splitLongMemoryParagraph(paragraph, MEMORY_MERGE_MAX_LENGTH)),
      MEMORY_MERGE_MAX_LENGTH
    );
    groups.forEach(group => {
      const idPart = `${turnInfo.messageIndexes?.[messagePosition] ?? messagePosition}:${role}:${group.start}-${group.end}`;
      const block = {
        messageIndex: turnInfo.messageIndexes?.[messagePosition] ?? messagePosition,
        idPart,
        paragraphIndex: group.start,
        paragraphEndIndex: group.end,
        role,
        speaker: isUser ? userName : (message.name || roleName),
        text: group.text
      };
      (isUser ? userBlocks : roleBlocks).push({
        ...block,
        text: isUser ? group.text : `角色卡：${group.text}`
      });
    });
  });

  const userText = userBlocks.map(block => block.text).filter(Boolean).join('\n\n');
  const userLine = userText ? `用户：${userText}` : '';
  const userIdPart = userBlocks.map(block => block.idPart).join('+');
  const sourceBlocks = roleBlocks.length > 0
    ? roleBlocks
    : userBlocks.map(block => ({ ...block, text: `用户：${block.text}` }));

  const turnFingerprint = getTurnFingerprint(turnInfo);
  return sourceBlocks.map((block, index) => {
    const includeUser = roleBlocks.length > 0 && Boolean(userLine);
    const paragraph = [includeUser ? userLine : '', block.text].filter(Boolean).join('\n');
    const idParts = [includeUser ? userIdPart : '', block.idPart].filter(Boolean).join('+');
    return {
      turn,
      sequence: index + 1,
      messageIndex: block.messageIndex,
      paragraphIndex: block.paragraphIndex,
      paragraphEndIndex: block.paragraphEndIndex,
      speaker: includeUser ? [userName, block.speaker].filter(Boolean).join(' + ') : block.speaker,
      role: includeUser ? 'mixed' : block.role,
      paragraph,
      summary: trimText(paragraph, 900),
      sourceText: [`第 ${turn || '?' } 轮`, paragraph].filter(Boolean).join('\n'),
      vectorChunkId: `${turn || 0}:${idParts}`,
      turnFingerprint,
      contentFingerprint: getContentFingerprint(paragraph)
    };
  });
}

export function buildFullTurnText(turnInfo) {
  const blocks = (turnInfo?.messages || []).map(message => {
    const text = cleanMessageText(message);
    if (!text) return '';
    return `${message.is_user === true || message.role === 'user' ? '用户' : '角色卡'}：${text}`;
  }).filter(Boolean);
  return blocks.join('\n\n').trim();
}

function bytesToBase64(bytes) {
  if (typeof btoa === 'function') {
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    return btoa(binary);
  }
  return Buffer.from(bytes).toString('base64');
}

function base64ToBytes(value) {
  if (typeof atob === 'function') {
    const binary = atob(String(value || ''));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }
  return new Uint8Array(Buffer.from(String(value || ''), 'base64'));
}

export function quantizeEmbedding(embedding) {
  if (!embedding || typeof embedding.length !== 'number' || embedding.length === 0) return null;
  let maxAbs = 0;
  for (let i = 0; i < embedding.length; i += 1) {
    const value = Math.abs(Number(embedding[i]) || 0);
    if (value > maxAbs) maxAbs = value;
  }
  if (maxAbs <= 0) return null;
  const quantized = new Int8Array(embedding.length);
  for (let i = 0; i < embedding.length; i += 1) {
    const scaled = Math.round(((Number(embedding[i]) || 0) / maxAbs) * 127);
    quantized[i] = Math.max(-127, Math.min(127, scaled));
  }
  return {
    embeddingQ: bytesToBase64(new Uint8Array(quantized.buffer)),
    embeddingScale: maxAbs / 127,
    embeddingDims: embedding.length,
    embeddingEncoding: 'int8:maxabs:v1'
  };
}

export function decodeQuantizedEmbedding(base64) {
  try {
    const bytes = base64ToBytes(base64);
    return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  } catch (_) {
    return new Int8Array();
  }
}

export function cosineSimilarity(a, b) {
  if (!a || !b || typeof a.length !== 'number' || typeof b.length !== 'number' || a.length === 0 || b.length === 0) return -1;
  const length = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < length; i += 1) {
    const av = Number(a[i]) || 0;
    const bv = Number(b[i]) || 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA === 0 || normB === 0) return -1;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const STOP_TERMS = new Set([
  '是不是', '有没有', '为什么', '怎么样', '怎么办', '什么', '这个', '那个',
  '还是', '还在', '还会', '了吗', '吗', '呢', '啊', '吧', '的', '了', '我', '你', '她', '他'
]);

export function extractQueryTerms(text) {
  const normalized = String(text || '').replace(/[^\p{Script=Han}A-Za-z0-9_]+/gu, ' ').trim();
  if (!normalized) return [];
  const terms = new Set();
  normalized.split(/\s+/).filter(Boolean).forEach(part => {
    if (/^[A-Za-z0-9_]{2,}$/.test(part)) {
      terms.add(part.toLowerCase());
      return;
    }
    const han = part.replace(/[^\p{Script=Han}]/gu, '');
    if (han.length >= 2) {
      for (let size = Math.min(4, han.length); size >= 2; size -= 1) {
        for (let i = 0; i <= han.length - size; i += 1) {
          const term = han.slice(i, i + size);
          if (!STOP_TERMS.has(term)) terms.add(term);
        }
      }
    } else if (han.length === 1 && !STOP_TERMS.has(han)) terms.add(han);
  });
  return Array.from(terms).filter(term => !STOP_TERMS.has(term)).sort((a, b) => b.length - a.length).slice(0, 20);
}

export function lexicalBoost(memory, queryTerms) {
  if (!Array.isArray(queryTerms) || queryTerms.length === 0) return { hits: 0, boost: 0, matched: [] };
  const text = `${memory?.sourceText || ''}\n${memory?.summary || ''}`.toLowerCase();
  const matched = queryTerms.filter(term => text.includes(term.toLowerCase()));
  return { hits: matched.length, boost: Math.min(0.08, matched.length * 0.015), matched };
}

export function sortByVectorScore(items) {
  return [...(Array.isArray(items) ? items : [])].sort((a, b) => {
    const scoreDiff = (Number(b.score) || 0) - (Number(a.score) || 0);
    if (Math.abs(scoreDiff) > 0.0001) return scoreDiff;
    return (Number(b.memory?.turn) || 0) - (Number(a.memory?.turn) || 0);
  });
}

export function rankVectorCandidates(memories, queryVector, queryText, options = {}) {
  const threshold = Number(options.threshold ?? MEMORY_DEFAULT_SIMILARITY_THRESHOLD) / 100;
  const excludedTurnFingerprints = options.excludedTurnFingerprints instanceof Set
    ? options.excludedTurnFingerprints
    : new Set(options.excludedTurnFingerprints || []);
  const terms = extractQueryTerms(queryText);
  const scored = [];
  (Array.isArray(memories) ? memories : []).forEach(memory => {
    if (memory?.enabled === false) return;
    if (options.model && memory.embeddingModel && memory.embeddingModel !== options.model) return;
    if (memory?.turnFingerprint && excludedTurnFingerprints.has(memory.turnFingerprint)) return;
    const vector = memory.embedding || decodeQuantizedEmbedding(memory.embeddingQ);
    const raw = cosineSimilarity(queryVector, vector);
    if (!Number.isFinite(raw) || raw <= -1 || raw < threshold) return;
    const lexical = lexicalBoost(memory, terms);
    scored.push({ memory, rawScore: raw, score: raw + lexical.boost, lexical });
  });
  const sorted = sortByVectorScore(scored);
  const seen = new Set();
  return sorted.filter(item => {
    const fingerprint = getRecallFingerprint(item.memory);
    if (!fingerprint || seen.has(fingerprint)) return false;
    seen.add(fingerprint);
    return true;
  });
}

export function applyRerankResults(vectorItems, rerankRows, options = {}) {
  const items = Array.isArray(vectorItems) ? vectorItems : [];
  const topK = Math.max(1, Number(options.topK) || 10);
  const threshold = Number(options.threshold ?? 0.35);
  if (!Array.isArray(rerankRows)) return { items: items.slice(0, topK), usedFallback: true };
  const selected = rerankRows
    .filter(row => Number.isInteger(Number(row?.index)) && Number(row.relevance_score) >= threshold && items[Number(row.index)])
    .sort((a, b) => Number(b.relevance_score) - Number(a.relevance_score))
    .slice(0, topK)
    .map(row => ({
      ...items[Number(row.index)],
      rerankScore: Number(row.relevance_score),
      score: Number(row.relevance_score)
    }));
  return { items: selected, usedFallback: false };
}

export function sortByTime(items) {
  return [...(Array.isArray(items) ? items : [])].sort((a, b) => {
    const turnDiff = (Number(a?.turn) || Number.MAX_SAFE_INTEGER) - (Number(b?.turn) || Number.MAX_SAFE_INTEGER);
    if (turnDiff !== 0) return turnDiff;
    return (Number(a?.sequence) || 0) - (Number(b?.sequence) || 0);
  });
}

export function mergeSameTurnResults(scoredItems, turns = []) {
  const ordered = [...(Array.isArray(scoredItems) ? scoredItems : [])];
  const groups = new Map();
  ordered.forEach(item => {
    const turn = Number(item.memory?.turn) || 0;
    if (turn <= 0) return;
    if (!groups.has(turn)) groups.set(turn, []);
    groups.get(turn).push(item);
  });
  const repeated = new Set([...groups.entries()].filter(([, values]) => values.length >= 2).map(([turn]) => turn));
  if (repeated.size === 0) return ordered;

  const turnByFingerprint = new Map();
  (Array.isArray(turns) ? turns : []).forEach(turn => {
    const fp = getTurnFingerprint(turn);
    if (fp) turnByFingerprint.set(fp, turn);
  });
  const mergedTurns = new Set();
  const result = [];
  ordered.forEach(item => {
    const turn = Number(item.memory?.turn) || 0;
    if (!repeated.has(turn)) {
      result.push(item);
      return;
    }
    if (mergedTurns.has(turn)) return;
    mergedTurns.add(turn);
    const group = groups.get(turn) || [item];
    const best = [...group].sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0))[0] || item;
    const currentTurn = turnByFingerprint.get(best.memory?.turnFingerprint);
    let text = buildFullTurnText(currentTurn);
    if (!text) {
      const userParts = [];
      const roleParts = [];
      sortByTime(group.map(entry => entry.memory)).forEach(memory => {
        const memoryText = String(memory?.paragraph || memory?.summary || memory?.sourceText || '').trim();
        const marker = '\n角色卡：';
        const markerIndex = memoryText.indexOf(marker);
        if (markerIndex >= 0) {
          if (!userParts.length) userParts.push(memoryText.slice(0, markerIndex).trim());
          roleParts.push(memoryText.slice(markerIndex + marker.length).trim());
        } else if (memoryText) {
          roleParts.push(memoryText);
        }
      });
      text = [userParts.filter(Boolean).join('\n\n'), roleParts.filter(Boolean).join('\n\n') ? `角色卡：${roleParts.filter(Boolean).join('\n\n')}` : ''].filter(Boolean).join('\n\n');
    }
    result.push({
      ...best,
      memory: { ...best.memory, paragraph: text, summary: text, sourceText: text },
      score: Math.max(...group.map(entry => Number(entry.score) || 0)),
      rawScore: Math.max(...group.map(entry => Number(entry.rawScore) || 0)),
      vectorMergedTurn: true
    });
  });
  return result;
}

export function xmlEscape(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function buildRecallPrompt(items) {
  const fragments = (Array.isArray(items) ? items : []).map(item => {
    const memory = item.memory || item;
    const score = Number(item.rerankScore ?? item.score ?? item.vectorScore ?? 0);
    const turn = xmlEscape(memory.turn || 0);
    const similarity = xmlEscape(`${(score * 100).toFixed(1)}%`);
    const text = String(memory.paragraph || memory.summary || memory.sourceText || '').trim();
    if (!text) return '';
    return `  <memory_fragment turn="${turn}" similarity="${similarity}">\n    ${text}\n  </memory_fragment>`;
  }).filter(Boolean);
  if (fragments.length === 0) return '';
  return [
    '<role_memory_vector_recall>',
    '  <description>',
    '    以下内容是从往期对话记录中按当前输入检索出的相关记忆分片，并非全部历史。',
    '    请尽力理解这些分片之间的前因后果、人物关系和情绪延续，理清它们与当前对话的关联。',
    '    这些分片已按原对话时间顺序排列；它们不一定是今天或刚才发生的内容，请不要误当作当前现场，只把它们作为过往经历和关系背景参考。',
    '  </description>',
    fragments.join('\n'),
    '</role_memory_vector_recall>'
  ].join('\n');
}

export function isBatchLimitError(error) {
  const message = String(error?.message || error || '').toLowerCase();
  return /batch|limit|exceed|too many|max(imum)?|数量|上限|超出/.test(message);
}

export function clampSetting(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(min, Math.min(max, number)) : fallback;
}
