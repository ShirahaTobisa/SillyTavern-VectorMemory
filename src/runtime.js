import {
  MEMORY_MAX_TOP_K,
  MEMORY_MIN_TOP_K,
  DEFAULT_STRIP_TAGS,
  assembleTurnFragments,
  buildConversationTurns,
  buildRecallPrompt,
  applyRerankResults,
  clampSetting,
  cleanMessageText,
  decodeQuantizedEmbedding,
  getContentFingerprint,
  getTurnFingerprint,
  mergeSameTurnResults,
  quantizeEmbedding,
  rankVectorCandidates,
  setStripTagList,
  sortByTime,
  stripVectorMemoryCode,
  trimText
} from './pure.js';
import { requestEmbeddings, requestRerank, requestModelList, normalizeBaseUrl } from './api.js';

const DEFAULT_SETTINGS = Object.freeze({
  enabled: false,
  autoPatrol: false,
  stripTags: '',
  preset: 'siliconflow',
  baseUrl: 'https://api.siliconflow.cn',
  apiKey: '',
  model: 'Qwen/Qwen3-Embedding-8B',
  dimensions: 1024,
  batchSize: 8,
  similarityThreshold: 50,
  topK: 10,
  keepFloors: 50,
  rerankEnabled: false,
  rerankApiFormat: 'jina',
  rerankUrl: '',
  rerankKey: '',
  rerankModel: 'Qwen/Qwen3-Reranker-8B',
  recallThreshold: 30,
  rerankCandidates: 50,
  rerankThreshold: 0.35
});

const PRESETS = {
  siliconflow: {
    baseUrl: 'https://api.siliconflow.cn',
    model: 'Qwen/Qwen3-Embedding-8B'
  },
  dashscope: {
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'text-embedding-v4'
  },
  gemini: {
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    model: 'gemini-embedding-001'
  }
};

const state = {
  initialized: false,
  eventsBound: false,
  generationStartedAt: 0,
  pendingPatrol: false,
  patrolTimer: null,
  patrolAbort: null,
  patrolRunning: false,
  chatRef: null,
  runtimeIndex: null,
  turnFingerprintCache: new WeakMap(),
  rerankWarned: false,
  modelMismatchWarned: false,
  largeIndexWarned: false,
  autoError: false,
  lastRecall: null,
  appliedStripTags: null,
  ui: null,
  progress: { current: 0, total: 0 }
};

function getContext() {
  return globalThis.SillyTavern?.getContext?.() || {};
}

function getExtensionSettings() {
  const context = getContext();
  const settings = context.extensionSettings || context.extension_settings || {};
  if (!settings.vectorMemory || typeof settings.vectorMemory !== 'object') settings.vectorMemory = {};
  Object.entries(DEFAULT_SETTINGS).forEach(([key, value]) => {
    if (settings.vectorMemory[key] === undefined) settings.vectorMemory[key] = value;
  });
  return settings.vectorMemory;
}

function saveSettings() {
  const context = getContext();
  try { context.saveSettingsDebounced?.(); } catch (error) { console.warn('[VectorMemory] save settings failed', error); }
}

function showToast(message, type = 'info') {
  const toast = globalThis.toastr?.[type];
  if (typeof toast === 'function') toast(message);
  else console[type === 'error' ? 'error' : 'log'](`[VectorMemory] ${message}`);
}

function eventName(context, name) {
  const types = context.event_types || context.eventTypes || {};
  return types[name] || name;
}

function getChat() {
  const context = getContext();
  return Array.isArray(context.chat) ? context.chat : [];
}

function getMetadata() {
  const context = getContext();
  return context.chatMetadata && typeof context.chatMetadata === 'object' ? context.chatMetadata : null;
}

function readPersistedIndex() {
  const metadata = getMetadata();
  const value = metadata?.vector_memory;
  return value && typeof value === 'object'
    ? {
      version: 1,
      fragments: Array.isArray(value.fragments) ? value.fragments : [],
      emptyTurnFingerprints: Array.isArray(value.emptyTurnFingerprints) ? value.emptyTurnFingerprints : []
    }
    : { version: 1, fragments: [], emptyTurnFingerprints: [] };
}

function hydrateIndex() {
  const persisted = readPersistedIndex();
  state.runtimeIndex = {
    fragments: persisted.fragments.map(fragment => ({
      ...fragment,
      embedding: decodeQuantizedEmbedding(fragment.embeddingQ)
    })),
    emptyTurnFingerprints: [...new Set(persisted.emptyTurnFingerprints.filter(Boolean))]
  };
  state.modelMismatchWarned = false;
  return state.runtimeIndex;
}

function ensureRuntimeIndex() {
  const chat = getChat();
  if (state.chatRef !== chat || !state.runtimeIndex) {
    state.chatRef = chat;
    state.turnFingerprintCache = new WeakMap();
    return hydrateIndex();
  }
  return state.runtimeIndex;
}

function serializableFragment(fragment) {
  const copy = { ...fragment };
  delete copy.embedding;
  return copy;
}

async function saveIndex() {
  const metadata = getMetadata();
  if (!metadata || !state.runtimeIndex) return;
  metadata.vector_memory = {
    version: 1,
    fragments: state.runtimeIndex.fragments.map(serializableFragment),
    emptyTurnFingerprints: [...new Set(state.runtimeIndex.emptyTurnFingerprints.filter(Boolean))]
  };
  const context = getContext();
  try {
    if (typeof context.saveMetadata === 'function') await context.saveMetadata();
  } catch (error) {
    console.warn('[VectorMemory] save metadata failed', error);
  }
}

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `vm-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function messageFingerprintCache(message) {
  if (!message || typeof message !== 'object') return '';
  const value = message.mes ?? message.content ?? '';
  const cached = state.turnFingerprintCache.get(message);
  if (cached && cached.value === value) return cached.fingerprint;
  const turn = { messages: [message] };
  const fingerprint = getTurnFingerprint(turn);
  state.turnFingerprintCache.set(message, { value, fingerprint });
  return fingerprint;
}

function cachedTurnFingerprint(turn) {
  // The turn fingerprint is cheap for the normal one-user/one-assistant case,
  // while the per-message cache prevents repeated cleaning during compression.
  const parts = [];
  (turn?.messages || []).forEach(message => {
    const clean = cleanMessageText(message);
    if (!clean) return;
    const marker = message.is_user === true || message.role === 'user' ? '用户：' : '角色卡：';
    parts.push(`${marker}${clean}`);
    messageFingerprintCache(message);
  });
  // getTurnFingerprint adds a role marker to objects, so compute directly via
  // a temporary user/assistant shaped turn to preserve the exact labels.
  const source = parts.join('\n').replace(/\s+/g, '').replace(/[，。、“”‘’：；！？,.!?;:"'`~]/g, '');
  return source.length >= 80 ? source.slice(0, 1000) : '';
}

function getTurnMarker(turn) {
  const fingerprint = cachedTurnFingerprint(turn);
  return fingerprint || `empty:${Number(turn?.turn) || 0}`;
}

function buildTurnFingerprintSet(chat, keepFloors) {
  const turns = buildConversationTurns(chat);
  if (!(Number(keepFloors) > 0) || chat.length === 0) return new Set();
  const start = Math.max(0, chat.length - Number(keepFloors));
  const retained = new Set();
  turns.forEach(turn => {
    if ((turn.messageIndexes || []).some(index => index >= start)) retained.add(getTurnMarker(turn));
  });
  return retained;
}

function applyKeepFloors(chat, index, keepFloors) {
  const keep = Number(keepFloors) || 0;
  if (!Array.isArray(chat) || keep <= 0 || chat.length <= keep) return { removed: 0, oldTurns: 0, covered: 0 };
  const start = chat.length - keep;
  const covered = new Set(index.fragments.map(fragment => fragment.turnFingerprint).filter(Boolean));
  const empty = new Set(index.emptyTurnFingerprints.filter(Boolean));
  const remove = new Set();
  let oldTurns = 0;
  let coveredTurns = 0;
  buildConversationTurns(chat).forEach(turn => {
    const indexes = turn.messageIndexes || [];
    if (indexes.length === 0 || indexes.some(indexValue => indexValue >= start)) return;
    oldTurns += 1;
    const marker = getTurnMarker(turn);
    if (covered.has(marker) || empty.has(marker)) {
      coveredTurns += 1;
      indexes.forEach(indexValue => remove.add(indexValue));
    }
  });
  [...remove].sort((a, b) => b - a).forEach(indexValue => chat.splice(indexValue, 1));
  return { removed: remove.size, oldTurns, covered: coveredTurns };
}

function isAllowedGenerationType(type) {
  return type === undefined || type === null || type === 'normal' || type === 'continue' || type === 'regenerate' || type === 'swipe';
}

function findLatestUser(chat) {
  for (let i = chat.length - 1; i >= 0; i -= 1) {
    const message = chat[i];
    if (message?.is_system === true || message?.extra?.vector_memory_injected === true) continue;
    if (message?.is_user === true || message?.role === 'user') return { message, index: i };
  }
  return null;
}

function modelMatches(fragment, model) {
  return fragment?.embeddingModel === model;
}

function warnModelMismatch(index, settings) {
  const mismatched = index.fragments.some(fragment => fragment.embeddingModel !== settings.model);
  const element = state.ui?.querySelector?.('[data-vm-model-warning]');
  if (element) element.hidden = !mismatched;
  if (mismatched && !state.modelMismatchWarned) {
    state.modelMismatchWarned = true;
    console.warn('[VectorMemory] fragments from another embedding model are excluded');
  }
  return mismatched;
}

async function retrieveMemories(chat, index, settings, abortSignal) {
  const latest = findLatestUser(chat);
  if (!latest) return [];
  const query = trimText(stripVectorMemoryCode(latest.message.mes ?? latest.message.content ?? ''), 800);
  if (!query) return [];
  const candidates = index.fragments.filter(fragment => {
    if (fragment.enabled === false || !modelMatches(fragment, settings.model)) return false;
    return fragment.embedding?.length || fragment.embeddingQ;
  });
  if (candidates.length === 0) return [];
  const keep = clampSetting(settings.keepFloors, 0, 80, 50);
  const excluded = buildTurnFingerprintSet(chat, keep);
  const threshold = settings.rerankEnabled
    ? clampSetting(settings.recallThreshold, 30, 100, 30)
    : clampSetting(settings.similarityThreshold, 50, 100, 50);
  const queryText = `当前问题：用户：${query}`;
  const [queryVector] = await requestEmbeddings([queryText], settings, { signal: abortSignal });
  const scored = rankVectorCandidates(candidates, queryVector, query, {
    threshold,
    model: settings.model,
    excludedTurnFingerprints: excluded
  });
  const topK = clampSetting(settings.topK, MEMORY_MIN_TOP_K, MEMORY_MAX_TOP_K, 10);
  let selected = scored.slice(0, settings.rerankEnabled
    ? clampSetting(settings.rerankCandidates, 20, 100, 50)
    : topK);

  if (settings.rerankEnabled && selected.length > 0) {
    try {
      const reranked = await requestRerank(query, selected.map(item => item.memory.summary || item.memory.paragraph || ''), {
        rerankApiFormat: settings.rerankApiFormat,
        rerankUrl: settings.rerankUrl
          || (settings.rerankApiFormat !== 'dashscope' && settings.baseUrl ? `${normalizeBaseUrl(settings.baseUrl)}/rerank` : ''),
        rerankKey: settings.rerankKey || settings.apiKey,
        rerankModel: settings.rerankModel,
        topN: topK
      }, { signal: abortSignal });
      const rerankResult = applyRerankResults(selected, reranked, {
        threshold: clampSetting(settings.rerankThreshold, 0, 1, 0.35),
        topK
      });
      selected = rerankResult.items;
    } catch (error) {
      if (error?.name === 'AbortError') return [];
      console.warn('[VectorMemory] rerank failed, using vector results', error);
      if (!state.rerankWarned) {
        state.rerankWarned = true;
        showToast('rerank 请求失败，已回退到向量结果', 'warning');
      }
      selected = selected.slice(0, topK);
    }
  }

  const merged = mergeSameTurnResults(selected, buildConversationTurns(chat));
  return sortByTime(merged);
}

function updateProgress(current, total) {
  state.progress = { current, total };
  const element = state.ui?.querySelector?.('[data-vm-progress]');
  if (element) element.textContent = total > 0 ? `${current}/${total}` : '';
}

function getSettingsSnapshot() {
  const settings = getExtensionSettings();
  settings.batchSize = clampSetting(settings.batchSize, 1, 16, 8);
  settings.topK = clampSetting(settings.topK, MEMORY_MIN_TOP_K, MEMORY_MAX_TOP_K, 10);
  settings.keepFloors = Number(settings.keepFloors) === 0
    ? 0
    : Math.max(30, Math.min(80, Math.round(clampSetting(settings.keepFloors, 30, 80, 50) / 2) * 2));
  settings.similarityThreshold = clampSetting(settings.similarityThreshold, 50, 100, 50);
  settings.recallThreshold = clampSetting(settings.recallThreshold, 30, 100, 30);
  settings.rerankCandidates = clampSetting(settings.rerankCandidates, 20, 100, 50);
  settings.rerankThreshold = clampSetting(settings.rerankThreshold, 0, 1, 0.35);
  if (state.appliedStripTags !== settings.stripTags) {
    setStripTagList(settings.stripTags);
    state.appliedStripTags = settings.stripTags;
    state.turnFingerprintCache = new WeakMap();
  }
  return settings;
}

// The flag self-heals: manual stops emit only GENERATION_STOPPED, and ST
// dry runs emit GENERATION_STARTED without a matching GENERATION_ENDED, so
// a latched boolean would stick forever.
function isGenerationActive() {
  if (!state.generationStartedAt) return false;
  if (Date.now() - state.generationStartedAt > 180_000) {
    console.warn('[VectorMemory] generation flag stale for 3min, clearing');
    state.generationStartedAt = 0;
    return false;
  }
  return true;
}

async function patrol(options = {}) {
  const settings = getSettingsSnapshot();
  const say = (message) => { if (options.interactive) showToast(message, 'warning'); };
  if (!settings.enabled) { say('请先勾选「启用向量记忆」'); return 0; }
  if (!options.interactive && !settings.autoPatrol) return 0;
  if (state.patrolRunning) { say('补录已在进行中'); return 0; }
  if (isGenerationActive()) {
    state.pendingPatrol = true;
    say('正在生成回复，结束后会自动补录');
    return 0;
  }
  const context = getContext();
  if (context.characterId === undefined) { say('群聊或未选择角色，暂不支持'); return 0; }
  const chat = getChat();
  if (!chat.length) { say('当前聊天还没有消息'); return 0; }
  const index = ensureRuntimeIndex();
  if (!options.interactive && index.fragments.length > 5000) {
    state.autoError = true;
    if (!state.largeIndexWarned) {
      state.largeIndexWarned = true;
      showToast('向量分片超过 5000 条，自动巡逻已暂停；可手动补录或重建索引', 'warning');
    }
    refreshUI();
    return 0;
  }

  state.patrolRunning = true;
  state.patrolAbort = new AbortController();
  refreshUI();
  const signal = state.patrolAbort.signal;
  const turns = buildConversationTurns(chat, { includeHidden: true });
  const settledTurns = options.full ? turns : turns.slice(0, Math.max(0, turns.length - 2));
  // Auto patrol only ingests the most recently settled turns; sweeping the
  // whole history is manual-only (user decision — RPH rescans everything).
  const eligibleTurns = (options.interactive || options.full) ? settledTurns : settledTurns.slice(-3);
  const existingByChunk = new Map(index.fragments.map(fragment => [fragment.vectorChunkId, fragment]));
  const existingFingerprints = new Set(index.fragments.map(fragment => fragment.contentFingerprint || getContentFingerprint(fragment.paragraph)).filter(Boolean));
  const pendingFingerprints = new Set();
  const fragmentItems = [];

  eligibleTurns.forEach(turn => {
    const turnFingerprint = getTurnFingerprint(turn);
    const marker = turnFingerprint || `empty:${turn.turn}`;
    const cleanHasContent = turn.messages.some(message => Boolean(cleanMessageText(message)));
    const fragments = assembleTurnFragments(turn, {
      userName: context.name1 || context.userName || '用户',
      roleName: context.name2 || 'AI'
    });
    if (!cleanHasContent || fragments.length === 0) {
      if (!index.emptyTurnFingerprints.includes(marker)) {
        index.emptyTurnFingerprints.push(marker);
      }
      return;
    }
    const emptyMarkers = [marker, `empty:${turn.turn}`];
    emptyMarkers.forEach(emptyMarker => {
      const emptyAt = index.emptyTurnFingerprints.indexOf(emptyMarker);
      if (emptyAt >= 0) {
        index.emptyTurnFingerprints.splice(emptyAt, 1);
      }
    });
    fragments.forEach(fragment => {
      const existing = existingByChunk.get(fragment.vectorChunkId);
      // A changed turn keeps the structural chunk id but gets a new
      // turn/content fingerprint, so it must be embedded again.
      if (existing && existing.turnFingerprint === fragment.turnFingerprint) return;
      if (fragment.contentFingerprint && (existingFingerprints.has(fragment.contentFingerprint) || pendingFingerprints.has(fragment.contentFingerprint))) return;
      if (fragment.contentFingerprint) pendingFingerprints.add(fragment.contentFingerprint);
      fragmentItems.push({ fragment, turn });
    });
  });

  updateProgress(0, fragmentItems.length);
  if (fragmentItems.length === 0 && options.interactive) {
    showToast('没有需要补录的新内容（最近 2 轮会等对话稳定后入库）', 'info');
  }
  let added = 0;
  let cursor = 0;
  let batchSize = clampSetting(settings.batchSize, 1, 16, 8);
  let failures = 0;
  let batchesSinceSave = 0;
  try {
    while (cursor < fragmentItems.length) {
      if (signal.aborted) throw Object.assign(new Error('Aborted'), { name: 'AbortError' });
      const batch = fragmentItems.slice(cursor, cursor + batchSize);
      try {
        const vectors = await requestEmbeddings(batch.map(item => item.fragment.sourceText), settings, { signal });
        vectors.forEach((vector, vectorIndex) => {
          const item = batch[vectorIndex];
          const packed = quantizeEmbedding(vector);
          if (!packed) return;
          const record = {
            id: makeId(),
            timestamp: Date.now(),
            turn: item.fragment.turn,
            turnFingerprint: item.fragment.turnFingerprint,
            summary: item.fragment.summary,
            paragraph: item.fragment.paragraph,
            sourceText: item.fragment.sourceText,
            contentFingerprint: item.fragment.contentFingerprint,
            vectorChunkId: item.fragment.vectorChunkId,
            sequence: item.fragment.sequence,
            enabled: true,
            embeddingModel: settings.model,
            embeddingDims: packed.embeddingDims,
            embeddingEncoding: packed.embeddingEncoding,
            embeddingScale: packed.embeddingScale,
            embeddingQ: packed.embeddingQ,
            embedding: new Int8Array(decodeQuantizedEmbedding(packed.embeddingQ))
          };
          index.fragments.push(record);
          existingByChunk.set(record.vectorChunkId, record);
          if (record.contentFingerprint) existingFingerprints.add(record.contentFingerprint);
          added += 1;
        });
        cursor += batch.length;
        failures = 0;
        batchesSinceSave += 1;
        updateProgress(cursor, fragmentItems.length);
        if (batchesSinceSave >= 4) {
          await saveIndex();
          batchesSinceSave = 0;
        }
      } catch (error) {
        if (error?.name === 'AbortError') throw error;
        failures += 1;
        const message = String(error?.message || '').toLowerCase();
        if (responseIsClientError(error) && /batch|limit|exceed|too many|max(imum)?|数量|上限|超出/.test(message) && batchSize > 1) {
          batchSize = Math.max(1, Math.floor(batchSize / 2));
          continue;
        }
        if (failures >= 3) throw error;
        await new Promise(resolve => setTimeout(resolve, 250));
      }
    }
    await saveIndex();
    state.autoError = false;
    if (options.interactive && fragmentItems.length > 0) showToast(`向量记忆补录完成：新增 ${added} 条`, 'success');
    return added;
  } catch (error) {
    await saveIndex();
    if (error?.name !== 'AbortError') {
      state.autoError = true;
      console.warn('[VectorMemory] patrol stopped', error);
      if (options.interactive) {
        showToast(`向量补录失败：${error.message || error}`, 'error');
        if (globalThis.confirm?.('向量补录连续失败，是否重试？')) {
          state.patrolRunning = false;
          state.patrolAbort = null;
          return patrol(options);
        }
      }
    }
    return added;
  } finally {
    state.patrolRunning = false;
    state.patrolAbort = null;
    refreshUI();
  }
}

function responseIsClientError(error) {
  return Number(error?.status) >= 400 && Number(error?.status) < 500
    || /(?:^|\D)4\d\d(?:\D|$)/.test(String(error?.message || error || ''))
    || /bad request|invalid request/.test(String(error?.message || '').toLowerCase());
}

function schedulePatrol(delay, options = {}) {
  if (state.patrolTimer) clearTimeout(state.patrolTimer);
  state.patrolTimer = setTimeout(() => {
    state.patrolTimer = null;
    patrol(options);
  }, delay);
}

function resetRuntimeForChat() {
  state.chatRef = null;
  state.runtimeIndex = null;
  state.turnFingerprintCache = new WeakMap();
  state.rerankWarned = false;
  state.modelMismatchWarned = false;
  state.largeIndexWarned = false;
  state.autoError = false;
  if (state.patrolAbort) state.patrolAbort.abort();
}

function bindEvents() {
  if (state.eventsBound) return;
  const context = getContext();
  const source = context.eventSource;
  if (!source || typeof source.on !== 'function') return;
  const on = (name, callback) => source.on(eventName(context, name), callback);
  const clearGenerationFlag = () => {
    state.generationStartedAt = 0;
    if (state.pendingPatrol) {
      state.pendingPatrol = false;
      // A deferred manual click was promised a follow-up, so resume as interactive.
      schedulePatrol(2000, { interactive: true });
    }
  };
  on('MESSAGE_RECEIVED', () => {
    clearGenerationFlag();
    if (getExtensionSettings().autoPatrol) schedulePatrol(2000);
  });
  on('CHAT_CHANGED', () => {
    resetRuntimeForChat();
    refreshUI();
  });
  on('GENERATION_STARTED', (_type, _params, dryRun) => {
    if (dryRun) return;
    state.generationStartedAt = Date.now();
    if (state.patrolAbort) state.patrolAbort.abort();
  });
  on('GENERATION_ENDED', clearGenerationFlag);
  on('GENERATION_STOPPED', clearGenerationFlag);
  on('MESSAGE_DELETED', () => refreshUI());
  state.eventsBound = true;
}

function settingsTemplate() {
  return `
    <div class="vm-settings" data-vector-memory-settings>
      <div class="inline-drawer">
        <div class="inline-drawer-toggle inline-drawer-header">
          <b>向量记忆 Vector Memory</b>
          <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
        </div>
        <div class="inline-drawer-content">
          <div class="vm-warning" data-vm-model-warning hidden>存在其它 embedding 模型的分片，当前配置不会召回它们；如需统一请重建索引。<button type="button" class="menu_button" data-vm-action="rebuild">重建索引</button></div>
          <div class="vm-warning" data-vm-native-warning hidden>检测到 SillyTavern 内置 Chat Vectorization 已启用，建议二选一。</div>
          <div class="vm-error" data-vm-auto-error hidden>自动巡逻已暂停，请检查 API 配置后手动补录。</div>
          <label class="checkbox_label"><input type="checkbox" data-vm-field="enabled"><span>启用向量记忆</span></label>
          <small class="vm-hint">开启后旧对话可整理成向量记忆；聊天很长时，旧楼层不再原文发送，改由按当前输入检索到的记忆分片代替，节省 token。</small>
          <label class="checkbox_label"><input type="checkbox" data-vm-field="autoPatrol"><span>自动录入新对话（RPH 原版行为）</span></label>
          <small class="vm-hint">打开后每次 AI 回复完成，只把最近完成的对话轮增量录入，不会去扫历史记录；历史旧楼层要录入请点下方「立即补录」。默认关闭 = 一切录入都手动。</small>
          <h4 class="vm-h">API 配置</h4>
          <div class="vm-field"><small>预设</small><select class="text_pole" data-vm-field="preset"><option value="siliconflow">SiliconFlow / 聚合器</option><option value="dashscope">DashScope (Qwen)</option><option value="gemini">Gemini (OpenAI compat)</option></select></div>
          <div class="vm-field"><small>Base URL</small><input class="text_pole" type="text" data-vm-field="baseUrl" placeholder="https://api.example.com"></div>
          <div class="vm-field"><small>API Key</small><input class="text_pole" type="password" data-vm-field="apiKey" autocomplete="off"></div>
          <div class="vm-field"><small>Embedding 模型</small>
            <div class="vm-inline"><select class="text_pole" data-vm-field="model"></select><button type="button" class="menu_button" data-vm-action="fetch-models">获取模型</button></div>
          </div>
          <div class="vm-actions"><button type="button" class="menu_button" data-vm-action="test">测试连接</button></div>
          <div class="inline-drawer vm-advanced">
            <div class="inline-drawer-toggle inline-drawer-header">
              <b>高级设置（默认即可，不用动）</b>
              <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
              <div class="vm-grid">
                <div class="vm-field"><small>维度</small><input class="text_pole" type="number" min="1" data-vm-field="dimensions" placeholder="1024"></div>
                <div class="vm-field"><small>批量大小</small><input class="text_pole" type="number" min="1" max="16" data-vm-field="batchSize"></div>
              </div>
              <div class="vm-field"><small>入库剔除标签（逗号分隔，支持 * 通配；改动后建议重建索引）</small><textarea class="text_pole" rows="2" data-vm-field="stripTags" placeholder="${DEFAULT_STRIP_TAGS}"></textarea></div>
              <small class="vm-hint">这些标签块（思维链、状态栏、变量更新、选项菜单等）整段不入库；留空 = 用上面的默认表。发现某张卡的怪标签漏进向量，就把标签名加进来（如 konatan_*）。</small>
              <div class="vm-field"><small>相似度阈值 <output data-vm-output="similarityThreshold"></output>%（低于此分数的记忆不召回）</small><input type="range" min="50" max="100" data-vm-field="similarityThreshold"></div>
              <div class="vm-grid">
                <div class="vm-field"><small>每次注入条数上限</small><input class="text_pole" type="number" min="10" max="20" data-vm-field="topK"></div>
                <div class="vm-field"><small>保留楼层数（0=不压缩）</small><input class="text_pole" type="number" min="0" max="80" step="2" data-vm-field="keepFloors"></div>
              </div>
              <label class="checkbox_label"><input type="checkbox" data-vm-field="rerankEnabled"><span>启用 rerank 重排（可选，进一步提升召回质量）</span></label>
              <div class="vm-grid">
                <div class="vm-field"><small>格式</small><select class="text_pole" data-vm-field="rerankApiFormat"><option value="jina">jina</option><option value="dashscope">dashscope</option></select></div>
                <div class="vm-field"><small>Rerank 模型</small><select class="text_pole" data-vm-field="rerankModel"></select></div>
              </div>
              <div class="vm-field"><small>Rerank URL（留空 = Base URL + /rerank）</small><input class="text_pole" type="text" data-vm-field="rerankUrl" placeholder="留空自动推导"></div>
              <div class="vm-field"><small>Rerank Key（留空 = 复用上方 API Key）</small><input class="text_pole" type="password" data-vm-field="rerankKey" autocomplete="off"></div>
              <div class="vm-field"><small>召回阈值 <output data-vm-output="recallThreshold"></output>%</small><input type="range" min="30" max="100" data-vm-field="recallThreshold"></div>
              <div class="vm-grid">
                <div class="vm-field"><small>候选数</small><input class="text_pole" type="number" min="20" max="100" data-vm-field="rerankCandidates"></div>
                <div class="vm-field"><small>rerank 阈值 <output data-vm-output="rerankThreshold"></output></small><input type="range" min="0" max="1" step="0.01" data-vm-field="rerankThreshold"></div>
              </div>
            </div>
          </div>
          <h4 class="vm-h">维护</h4>
          <div class="vm-stats-row">当前索引：<span data-vm-stats>0 条 / 0.0 KB</span> <span data-vm-progress></span></div>
          <div class="vm-stats-row">上次召回：<span data-vm-lastrecall>暂无（发消息生成时自动触发）</span></div>
          <div class="vm-actions">
            <button type="button" class="menu_button" data-vm-action="patrol">立即补录</button>
            <button type="button" class="menu_button" data-vm-action="trim">修剪未来分片</button>
            <button type="button" class="menu_button" data-vm-action="rebuild">重建索引</button>
            <button type="button" class="menu_button" data-vm-action="clear">清空索引</button>
          </div>
          <div class="vm-search-row"><input class="text_pole" type="search" data-vm-search placeholder="输入要查的事实"><button type="button" class="menu_button" data-vm-action="search">检索</button></div>
          <div data-vm-results></div>
        </div>
      </div>
    </div>`;
}

function setFieldValue(element, value) {
  if (!element) return;
  if (element.type === 'checkbox') element.checked = Boolean(value);
  else element.value = value ?? '';
}

function readFieldValue(element) {
  if (element.type === 'checkbox') return element.checked;
  if (element.type === 'number' || element.type === 'range') return Number(element.value);
  return element.value;
}

function estimateIndexSize(index) {
  try { return new TextEncoder().encode(JSON.stringify({ fragments: index.fragments.map(serializableFragment), emptyTurnFingerprints: index.emptyTurnFingerprints })).length; } catch (_) { return JSON.stringify(index).length; }
}

function ensureSelectOption(select, value) {
  if (!select || select.tagName !== 'SELECT') return;
  const target = String(value ?? '');
  if (![...select.options].some(option => option.value === target)) {
    const option = document.createElement('option');
    option.value = target;
    option.textContent = target || '（未选择，点「获取模型」）';
    select.insertBefore(option, select.firstChild);
  }
}

function fillModelSelect(select, models, current) {
  if (!select) return;
  const value = String(current || select.value || '');
  select.innerHTML = '';
  const ids = [...(Array.isArray(models) ? models : [])];
  if (value && !ids.includes(value)) ids.unshift(value);
  ids.forEach(id => {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = id;
    select.appendChild(option);
  });
  ensureSelectOption(select, value);
  select.value = value;
}

function refreshUI() {
  if (!state.ui) return;
  const settings = getSettingsSnapshot();
  ensureSelectOption(state.ui.querySelector('[data-vm-field="model"]'), settings.model);
  ensureSelectOption(state.ui.querySelector('[data-vm-field="rerankModel"]'), settings.rerankModel);
  state.ui.querySelectorAll('[data-vm-field]').forEach(element => setFieldValue(element, settings[element.dataset.vmField]));
  state.ui.querySelectorAll('[data-vm-output]').forEach(element => {
    const value = settings[element.dataset.vmOutput];
    element.textContent = element.dataset.vmOutput === 'rerankThreshold' ? Number(value).toFixed(2) : String(value ?? '');
  });
  const index = ensureRuntimeIndex();
  const bytes = estimateIndexSize(index);
  const stats = state.ui.querySelector('[data-vm-stats]');
  if (stats) stats.textContent = `${index.fragments.length} 条 / ${(bytes / 1024).toFixed(1)} KB`;
  const autoError = state.ui.querySelector('[data-vm-auto-error]');
  if (autoError) autoError.hidden = !state.autoError;
  const patrolButton = state.ui.querySelector('[data-vm-action="patrol"]');
  if (patrolButton) {
    patrolButton.disabled = state.patrolRunning;
    patrolButton.textContent = state.patrolRunning ? '补录中…' : '立即补录';
  }
  const nativeWarning = state.ui.querySelector('[data-vm-native-warning]');
  const extensionSettings = getContext().extensionSettings || getContext().extension_settings || {};
  if (nativeWarning) nativeWarning.hidden = extensionSettings.vectors?.enabled_chats !== true;
  const lastRecallElement = state.ui.querySelector('[data-vm-lastrecall]');
  if (lastRecallElement && state.lastRecall) {
    const recall = state.lastRecall;
    const detail = recall.count > 0
      ? `注入 ${recall.count} 条（轮 ${[...new Set(recall.turns)].join(',')}｜分 ${recall.scoreRange}）`
      : '注入 0 条（无匹配或低于阈值）';
    lastRecallElement.textContent = `${new Date(recall.at).toLocaleTimeString()} ${detail}，压缩 ${recall.removedFloors} 楼（保留区外已覆盖 ${recall.coverage} 轮）`;
  }
  warnModelMismatch(index, settings);
}

async function manualSearch(query) {
  const settings = getSettingsSnapshot();
  const index = ensureRuntimeIndex();
  const cleanQuery = trimText(stripVectorMemoryCode(query), 800);
  const resultsElement = state.ui?.querySelector?.('[data-vm-results]');
  if (!resultsElement || !cleanQuery) return;
  try {
    const context = getContext();
    const latest = getChat().findLast?.(message => message?.is_user === true) || { is_user: true, mes: cleanQuery };
    const signal = new AbortController().signal;
    const results = await retrieveMemories([...getChat(), { ...latest, mes: cleanQuery }], index, settings, signal);
    resultsElement.innerHTML = results.length
      ? results.map(item => `<div class="vm-result"><b>第 ${item.memory.turn} 轮 / ${(Number(item.rerankScore ?? item.score ?? 0) * 100).toFixed(1)}%</b><div>${escapeHtml(item.memory.paragraph || '')}</div></div>`).join('')
      : '<div class="vm-empty">没有找到匹配分片</div>';
    void context;
  } catch (error) {
    resultsElement.textContent = error.message || '检索失败';
  }
}

function escapeHtml(value) {
  return String(value || '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[character]));
}

function bindUI() {
  if (state.ui) return;
  const host = document.querySelector('#extensions_settings, #extensions_settings2');
  if (!host) return;
  const wrapper = document.createElement('div');
  wrapper.innerHTML = settingsTemplate();
  state.ui = wrapper.firstElementChild;
  host.appendChild(state.ui);
  state.ui.querySelectorAll('[data-vm-field]').forEach(element => {
    element.addEventListener('change', () => {
      const settings = getExtensionSettings();
      const key = element.dataset.vmField;
      const value = readFieldValue(element);
      if (key === 'preset' && PRESETS[value]) {
        settings.preset = value;
        settings.baseUrl = PRESETS[value].baseUrl;
        settings.model = PRESETS[value].model;
        saveSettings();
        refreshUI();
      } else {
        settings[key] = value;
        saveSettings();
      }
      refreshUI();
    });
    element.addEventListener('input', () => {
      if (element.type === 'range') {
        const settings = getExtensionSettings();
        settings[element.dataset.vmField] = readFieldValue(element);
        saveSettings();
        refreshUI();
      }
    });
  });
  state.ui.querySelector('[data-vm-action="patrol"]')?.addEventListener('click', () => patrol({ interactive: true }));
  const rebuild = async () => {
    if (!globalThis.confirm?.('重建索引会清空当前聊天的向量分片，继续吗？')) return;
    const index = ensureRuntimeIndex();
    index.fragments = [];
    index.emptyTurnFingerprints = [];
    await saveIndex();
    await patrol({ interactive: true, full: true });
  };
  state.ui.querySelectorAll('[data-vm-action="rebuild"]').forEach(button => button.addEventListener('click', rebuild));
  // Same semantics as RPH 1.7.9 story branches: trim by turn number, never by
  // fingerprint — a cleaning-rule change must not make the whole index look
  // deletable.
  state.ui.querySelector('[data-vm-action="trim"]')?.addEventListener('click', async () => {
    const index = ensureRuntimeIndex();
    const maxTurn = buildConversationTurns(getChat(), { includeHidden: true }).length;
    const doomed = index.fragments.filter(fragment => Number(fragment.turn) > maxTurn);
    if (doomed.length === 0) {
      showToast(`没有未来分片（当前对话共 ${maxTurn} 轮）`, 'info');
      return;
    }
    if (!globalThis.confirm?.(`将删除 ${doomed.length} 条轮号超出当前对话（共 ${maxTurn} 轮）的分片，通常来自开分支前的未来剧情。继续吗？`)) return;
    index.fragments = index.fragments.filter(fragment => Number(fragment.turn) <= maxTurn);
    await saveIndex();
    refreshUI();
    showToast(`已修剪 ${doomed.length} 条未来分片`, 'success');
  });
  state.ui.querySelector('[data-vm-action="clear"]')?.addEventListener('click', async () => {
    if (!globalThis.confirm?.('清空当前聊天的向量索引，继续吗？')) return;
    const index = ensureRuntimeIndex();
    index.fragments = [];
    index.emptyTurnFingerprints = [];
    await saveIndex();
    refreshUI();
  });
  state.ui.querySelector('[data-vm-action="fetch-models"]')?.addEventListener('click', async event => {
    const button = event.currentTarget;
    const settings = getSettingsSnapshot();
    try {
      button.disabled = true;
      const models = await requestModelList(settings);
      const embedModels = models.filter(id => /embed|bge|gte|\be5\b|e5-/i.test(id));
      const rerankModels = models.filter(id => /rerank/i.test(id));
      fillModelSelect(state.ui.querySelector('[data-vm-field="model"]'), embedModels.length ? embedModels : models, settings.model);
      fillModelSelect(state.ui.querySelector('[data-vm-field="rerankModel"]'), rerankModels.length ? rerankModels : models, settings.rerankModel);
      showToast(`获取到 ${models.length} 个模型，已过滤出 embedding 候选 ${embedModels.length} 个`, 'success');
    } catch (error) {
      showToast(`获取模型失败：${error.message || error}`, 'error');
    } finally {
      button.disabled = false;
    }
  });
  state.ui.querySelector('[data-vm-action="test"]')?.addEventListener('click', async event => {
    const button = event.currentTarget;
    const settings = getSettingsSnapshot();
    try {
      button.disabled = true;
      const [vector] = await requestEmbeddings(['连接测试'], settings);
      showToast(`连接成功：${settings.model}，维度 ${vector.length}`, 'success');
    } catch (error) {
      showToast(`连接失败：${error.message || error}`, 'error');
    } finally {
      button.disabled = false;
    }
  });
  state.ui.querySelector('[data-vm-action="search"]')?.addEventListener('click', () => manualSearch(state.ui.querySelector('[data-vm-search]')?.value || ''));
  state.ui.querySelector('[data-vm-search]')?.addEventListener('keydown', event => {
    if (event.key === 'Enter') manualSearch(event.currentTarget.value);
  });
  refreshUI();
}

export async function vectorMemoryInterceptor(chat, contextSize, abort, type) {
  if (!isAllowedGenerationType(type) || !Array.isArray(chat)) return;
  const settings = getSettingsSnapshot();
  if (!settings.enabled) return;
  const context = getContext();
  if (context.characterId === undefined) return;
  const index = ensureRuntimeIndex();
  if (index.fragments.length === 0) return;
  const signal = abort?.signal || abort;

  // Only splice the prompt array. Existing chat message objects are never
  // edited, because ST writes those objects back to the chat file.
  const removedFloors = applyKeepFloors(chat, index, settings.keepFloors);
  if (signal?.aborted) return;
  warnModelMismatch(index, settings);
  try {
    const selected = await retrieveMemories(chat, index, settings, signal);
    const scores = selected.map(item => Number(item.rerankScore ?? item.score ?? 0));
    state.lastRecall = {
      at: Date.now(),
      count: selected.length,
      turns: selected.map(item => Number(item.memory?.turn) || 0),
      scoreRange: scores.length ? `${Math.min(...scores).toFixed(2)}~${Math.max(...scores).toFixed(2)}` : '',
      removedFloors: removedFloors.removed,
      coverage: `${removedFloors.covered}/${removedFloors.oldTurns}`
    };
    console.info('[VectorMemory] recall', state.lastRecall, selected.map(item => ({
      turn: item.memory?.turn,
      score: Number(item.rerankScore ?? item.score ?? 0).toFixed(3),
      text: String(item.memory?.paragraph || '').slice(0, 80)
    })));
    refreshUI();
    if (!selected.length) return;
    const latest = findLatestUser(chat);
    if (!latest) return;
    const prompt = buildRecallPrompt(selected);
    if (!prompt) return;
    const username = context.name1 || context.userName || 'User';
    chat.splice(latest.index, 0, {
      is_user: true,
      name: username,
      send_date: Date.now(),
      mes: prompt,
      extra: { vector_memory_injected: true }
    });
  } catch (error) {
    if (error?.name !== 'AbortError') console.warn('[VectorMemory] recall failed; generation continues without injection', error);
  }
}

export function initVectorMemory() {
  if (state.initialized) return;
  state.initialized = true;
  getSettingsSnapshot();
  bindEvents();
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindUI, { once: true });
    else bindUI();
  }
  ensureRuntimeIndex();
}

export const vectorMemoryInternals = {
  state,
  getContext,
  getSettingsSnapshot,
  ensureRuntimeIndex,
  patrol,
  applyKeepFloors,
  retrieveMemories,
  refreshUI
};
