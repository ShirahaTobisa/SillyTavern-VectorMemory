# Vector Memory (RPH)

SillyTavern 1.18.0 的第三方向量记忆扩展。它把稳定滞后的对话轮清洗、分段、量化入库，并在生成前通过 embedding + lexical boost（可选 rerank）召回相关历史，再用 `keepFloors` 从本次 prompt 中置换已覆盖的旧轮。

## 安装

1. 将本仓库推到自己的 GitHub（本地仓库已经 `git init`，不会自动 push）。
2. 在 SillyTavern 打开 **Extensions → Install extension**。
3. 填写仓库 URL，确认安装并重载页面。

本地开发也可以把整个目录放到 `scripts/extensions/third-party/`。`manifest.json` 的相对 `index.js`、`style.css` 和 `src/` 路径与 Install extension 安装一致。

## 设置

- 总开关默认关闭。关闭时不巡逻、不压缩、不注入。
- 「自动录入新对话」默认关闭：关闭时一切录入都靠「立即补录」。打开后仅在每次 AI 回复完成后把最近完成的对话轮增量录入（滑动窗口 3 轮 + 去重），永远不会自动扫历史记录；历史旧楼层的全量补录只能手动。
- 基础配置只有四项：预设、Base URL、API Key、Embedding 模型。模型点「获取模型」从 `GET /models` 拉列表下拉选择（自动过滤出 embedding 类模型），「测试连接」一键验证配置并报告维度。
- 维度/批量/相似度阈值/注入条数/保留楼层和整个 rerank 区都收在「高级设置」折叠里，默认值即推荐值（维度 1024、批量 8、阈值 50、Top K 10、保留 50 层，保留楼层设 0 可关闭压缩）。
- rerank 默认关闭。URL 留空自动用 Base URL + `/rerank`，Key 留空复用上方 API Key；`jina` 格式默认模型 `Qwen/Qwen3-Reranker-8B`，阈值默认 0.35，不同模型的分数标定不同，需要自行调整。
- 「入库剔除标签」（高级设置）：思维链、状态栏、变量更新、选项菜单、生图提示（`image###...###`）等标签块整段不入库，支持 `*` 通配（如 `konatan_*`），并处理 prefill 只留闭合标签、消息截断只留开标签的孤儿情况。留空使用内置默认表；发现某张卡的怪标签漏进向量就把标签名加进列表，改动后建议重建索引。
- DashScope 和 Gemini 预设会填入完整兼容路径；改 embedding 模型或维度后需要手动重建索引，旧分片不会自动清理。
- 维护区可查看当前聊天的条数与元数据体积、立即补录、重建/清空索引，以及复用完整召回链路的手动检索。

索引保存到当前聊天的 `chatMetadata.vector_memory`，全局参数保存到 `extension_settings.vectorMemory`。切换聊天会重建运行态解码向量。检测到 SillyTavern 内置 Chat Vectorization 时会显示二选一提示。

## 限制

只支持单角色非群聊；群聊会完全旁路。`quiet`/`impersonate` 生成旁路，不做主动工具检索、classic 总结、本地 embedding、World Info/Data Bank 联动，也不读取 ST 内置 vectors 数据。消息编辑或删除不会自动清理孤儿分片，积累较多时请重建索引。索引超过 5000 条后自动巡逻暂停，手动补录仍可执行。

生成拦截器只对传入的 chat 数组做 splice，绝不修改消息对象本身，因此注入内容不会写回真实聊天记录。

## 自测

仓库不需要 SillyTavern 运行时即可测试纯函数和请求层：

```bash
npm test
```

测试覆盖清洗、1800/400 边界、fragment 组装、指纹去重、int8 量化/解码余弦误差、lexical boost、同轮合并、embedding 响应排序、批量降半和 rerank 失败回退。
