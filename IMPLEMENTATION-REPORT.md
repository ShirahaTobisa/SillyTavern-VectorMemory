# Implementation Report

日期：2026-08-05

## 实现范围

- `src/pure.js`：清洗、轮次识别、1800/400 分段与贪心合并、fragment 组装、指纹、int8 maxabs 量化/解码、余弦、词法 boost、排序、同轮合并和注入文案。
- `src/api.js`：OpenAI 兼容 embedding、Jina/DashScope rerank、60 秒超时、可注入 `fetch`、响应校验和 4xx 批量降半。
- `src/runtime.js`：ST context 接入、per-chat 元数据、巡逻调度、keepFloors splice 压缩、模型一致性守卫、召回和设置面板。
- `index.js`：经典脚本入口通过相对动态 import 加载模块，并提供 manifest 要求的全局 `vectorMemoryInterceptor`。

## 与规格的偏差和处置

1. 规格同时要求“baseUrl 末尾没有 `/v1` 就自动补 `/v1`”和 Gemini 预设 `.../v1beta/openai`。Gemini 实际兼容端点不接受再追加 `/v1`，因此 `normalizeBaseUrl()` 对包含 `/v1`、`/v1beta` 或以 `/openai` 结尾的完整兼容路径保持原样，普通地址仍追加 `/v1`。这是保证两个定案预设可工作的最小处置。
2. 规格规定短轮 `turnFingerprint` 为 `''`，同时要求空轮把该指纹写入 `emptyTurnFingerprints` 并可用于压缩。空字符串无法区分不同空轮；实现以 `empty:<turn>` 作为仅用于空轮表和压缩判定的内部标记，非空短轮仍保持空 `turnFingerprint`、不可覆盖。
3. 规格的双重去重文字写成“命中 `vectorChunkId` 跳过”，而消息编辑条款又要求同一结构位置、内容指纹变化时新增。实现只在 chunk id 且旧 fragment 的 `turnFingerprint` 相同才跳过；指纹变化会新增孤儿分片，符合编辑安全条款。
4. `settings.html` 作为 manifest 交付文件保留，但设置模板由 `runtime.js` 注入 Extensions 抽屉；这是规格允许的“settings.html 或 JS 模板字符串二选一”，未依赖 ST 内部模块。
5. 本工作区没有可启动的 SillyTavern 1.18.0 实例，因此无法做真实端点和真实 UI 安装验收。已完成 Node 纯函数/API 自测、入口加载检查，以及用最小 `getContext()` stub 验证压缩、注入、群聊/quiet 门控和原消息不变性。
6. 为兼容不同 1.18 构建，运行时在 `getContext()` 内同时接受 `extensionSettings`/`extension_settings`、`event_types`/`eventTypes`，事件名缺失时使用规格中的字符串；没有导入 `/script.js` 或任何 ST 内部路径。

## 自测命令

在仓库根执行：

```text
npm test
node --check index.js
node --check src/pure.js
node --check src/api.js
node --check src/runtime.js
git diff --check
```

结果：13 个 Node 测试全部通过；四个 JS 文件语法检查通过；无 whitespace 错误。

## §10 验收清单自查

1. **60+ 轮入库/幂等**：轮次、稳定滞后、≤400 分组、`用户：/角色卡：`、`第 N 轮`、量化字段和双重去重均在 `patrol()`/`assembleTurnFragments()`；纯函数测试覆盖组装和去重。真实 ST 补录尚未执行。
2. **代码/HTML 清洗**：`stripVectorMemoryCode()` 按 RPH 顺序实现七条逐行启发式；Node 测试验证正常文本保留、代码围栏、script 和 HTML 清除。
3. **keepFloors/注入/不写回**：stub 运行验证旧轮 splice、注入位置和原对象文本不变；实现只对传入 chat 数组 splice。
4. **压缩区事实召回**：候选使用 int8 余弦、词法 boost、模型过滤和 query 前缀；未连接真实 embedding 服务，需用户实端点复核答案质量。
5. **同轮合并**：`mergeSameTurnResults()` 有独立测试，优先取当前轮完整清洗文本，缺失时按 fragment fallback。
6. **保留区排除**：`buildTurnFingerprintSet()` 在召回前建立保留区指纹集合，候选过滤使用 turnFingerprint；keepFloors=0 时不排除。
7. **rerank/500 回退**：Jina 请求、阈值和排序有 mock 测试；HTTP 失败由 runtime `console.warn`、每会话一次 toast 并回退向量结果。
8. **Qwen/Gemini/降批**：三种 URL/模型预设已写入设置，Gemini 路径保留规则有测试；真实 key/端点未执行；批量 limit mock 已验证 2→1 降半重试。
9. **quiet/impersonate/总开关**：类型门控和 `enabled` 早退已实现；stub 验证 quiet 与群聊旁路。
10. **切换聊天/群聊隔离**：`CHAT_CHANGED` abort 并清空运行态，随后从当前 `chatMetadata` 重建；只读 `chatMetadata` 不长期持有引用；群聊按 `characterId === undefined` 旁路。
11. **编辑安全**：turnFingerprint 从当前清洗文本实时计算；失配轮不会被 keepFloors 删除，巡逻会在结构 id 相同但指纹变化时新增分片。真实编辑流程需在 ST 实例复核。
12. **仓库形态**：manifest 字段齐全、README/LICENSE/.gitignore/tests 存在，未 push；本报告完成后提交单一干净 commit，提交前再次确认 `git status`。
