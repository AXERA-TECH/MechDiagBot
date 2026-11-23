# 架构说明

## 设计约束

1. LLM 负责语言理解、检索编排和解释，不直接控制设备。
2. 状态事实优先来自设备接口，已知查询不经过生成模型。
3. 诊断工具默认只读，并通过显式白名单注册。
4. 一个 Session 同时只执行一个 Turn，避免上下文和确认流程串线。
5. Agent Runner 必须设置模型步数上限。
6. 所有重要过程写成追加事件，便于回放、评测和审计。
7. 具体供应商技术只能出现在 Adapter 或组合根中。

## 边界

### 外层 AgentLoop

`AgentLoop` 是一次用户交互的生命周期和持久化边界；实际任务由 `TaskOrchestrator` 编排：

```text
turn/start
  → user/utterance
  → telemetry snapshot
  → plan/created
  → plan/step_started / plan/step_completed
  → assistant/response
turn/completed
```

每个 Session 使用独立异步锁。不同 Session 可以并行，同一 Session 串行执行。

### 内层 BoundedAgentRunner

Runner 只服务诊断通道：

```text
model request
  ├── final answer → return
  └── tool calls
        → execute approved tools
        → append tool results
        → next model step
```

达到 `max_steps` 后返回安全降级响应，不继续自主循环。

### 上下文与记忆

当前实现将“设备事实、会话历史、长期记忆和审计事件”分开处理：

```text
MachineTelemetryPort ──→ MachineSnapshot（当前权威事实）
SessionStorePort      ──→ 最近会话 Turn + 可选摘要
MemoryPort            ──→ 按机器/操作者隔离的显式记忆
EventStorePort        ──→ 不可变审计事件，不自动回灌 Prompt
                              ↓
                       ContextBuilder
                              ↓
                       BoundedAgentRunner
```

`ContextBuilder` 每轮只加载有限数量的最近 Turn，并应用字符预算；旧历史优先丢弃，当前机器快照、系统安全规则和当前请求优先保留。记忆会被标记为“非权威设备状态”，不能覆盖实时遥测。

默认演示程序将会话 Turn 和显式记忆分别追加到：

```text
runtime/sessions.jsonl
runtime/memory.jsonl
```

它们与 `runtime/events.jsonl` 审计日志分离。`JsonlSessionStore` 和 `JsonlMemoryStore` 只是无第三方依赖的演示实现，生产环境应替换为 SQLite 或独立存储服务。当前不会自动把模型推测写入长期记忆；记忆必须由受信任流程显式追加。

宿主可通过 `AgentLoop.remember()` 写入经过校验的 `MemoryRecord`，通过 `set_session_summary()` 写入外部摘要器生成的摘要；二者都不会由 LLM 自主调用。

### 任务编排与策略执行

请求首先由 `TaskPlanner` 拆成 `TaskPlan`，再由 `PlanExecutor` 按依赖关系执行。典型步骤包括：

```text
observe → retrieve → infer → propose → approve → act → verify
```

每个步骤引用一个 `CapabilityDescriptor`。能力描述包含副作用、风险等级、是否需要审批和来源（内置、插件、MCP 或 Skill）。`PolicyEngine` 在执行前确定性地校验这些属性；LLM 可以提出计划，但不能批准计划或绕过策略。

需要审批的计划以 `PendingApproval` 暂存，绑定发起人 `operator_id`、目标 `machine_id`、计划摘要和生成时的设备状态签名，默认 5 分钟过期。确认时必须同时匹配发起人、设备与当前状态；否定或取消语义优先识别，取消、超时、换人、换设备、设备状态变化或发起新请求都会阻止旧计划执行。每次状态迁移写入 `approval/*` 审计事件。

当前面向少量能力的默认生产路线是直接 Tool Call：`ToolCallingTaskPlanner` 把所有已启用 Capability 的名称、描述和输入 Schema 投影成模型 Function Tools，模型选择最小调用集合并给出参数。调用不会直接执行，而是先转换成顺序依赖的 `TaskPlan`：

```text
enabled CapabilityDescriptor
  → Function Tool projection
  → LLM Tool Calls
  → TaskStep DAG
  → PolicyEngine
  → CapabilityRegistry.execute()
```

模型无响应、没有选择能力或返回未知工具时，规划会退回 `RuleBasedTaskPlanner`。当能力数量增长到几十个以上，再在 Function Tool projection 前加入候选检索；执行合同不需要变化。

例如“查看当前状态，查询 E123 手册，给出方案，确认后停机”会形成：

```text
telemetry.read → telemetry.summarize → knowledge.search
→ diagnostic.agent → action.propose → approval → machine.control → verify.state
```

`machine.control` 仍由 Safety Gateway 占位执行器承接，未经审批不会产生设备写入。审批通过后执行器写回内存遥测，`verify.state` 重新读取实时遥测并与 `action_result` 的预期比对：一致时报告动作已生效，不一致时明确提示人工检查。否定语气、疑问句和启动/停止冲突的指令在规划器与执行器两处都会被拒绝。

### 能力 Provider 与 Artifact

所有可扩展能力实现统一的 `CapabilityProvider` 合同：

```text
CapabilityProvider
  ├── descriptor        名称、版本、Schema、风险、来源、触发词
  ├── execute()         执行并返回 CapabilityResult
  ├── startup()         可选启动钩子，由 Registry 调用
  ├── shutdown()        可选关闭钩子，由 Registry 调用
  └── health()          可选健康检查
```

`CapabilityRegistry` 负责注册、启停、按 owner 卸载、健康检查、输入/输出 Schema 校验、超时和机型适配检查。Plugin 和后续 MCP Host 都注册 Provider，不修改 `AgentLoop` 或 `PlanExecutor`。

能力的 `enabled` 与 `started` 是两个状态：注册时可以是 enabled，但不会在同步注册函数里创建连接；应用启动时 Registry 的 `start_all()`，或首次执行前的启动兜底，会调用 Provider 的 `startup()`，并保证每个 Provider 只启动一次。插件卸载先关闭其 Provider，再移除 owner；应用关闭会继续关闭所有 core Provider，即使某个关闭钩子失败也不会跳过其他资源。

模型、内层只读工具和整轮请求有独立的应用层 deadline。分别由 `BoundedAgentRunner`、`ToolRegistry` 和 `AgentLoop` 取消超时任务并记录事件；Capability 自身还受 Descriptor 的执行超时约束。

步骤之间不传递任意 Python 返回值或拼接文本，而是传递 `Artifact`：

```text
Artifact
  ├── artifact_type
  ├── data
  ├── source_capability
  ├── confidence
  ├── provenance
  └── observed_at
```

例如，操作员说“指示灯亮、报警声响”时，当前内置 `observation.user_report` 会生成低置信度的 `indicator_report` 和 `audio_alarm_report`，明确标记为操作员报告而非传感器事实。未来安装真实视觉或音频 Provider 后，可以注册相同领域含义但更高置信度、带传感器来源的 Artifact。

Planner 不包含 `if capability == indicator.inspect` 之类的分支。观察类 Provider 可通过 `auto_select` 和 `trigger_terms` 声明发现条件；新增视觉、音频或传感器 Provider 后，Planner 会自动将其插入证据链。

## 视觉扩展

视觉不会进入领域核心。后续新增：

```text
VisionPort                  # 稳定接口
DeepStreamVisionAdapter     # 摄像头/推理实现
VisionSnapshotTool          # Agent 只读工具
VisionEventAdapter          # 将结构化检测写入事件流
```

建议接口：

```python
class VisionPort(Protocol):
    async def get_snapshot(self, camera: str) -> ImageRef: ...
    async def get_recent_events(self, event_type: str) -> Sequence[VisionEvent]: ...
    async def inspect_component(self, component: str) -> InspectionResult: ...
```

接入步骤：

1. 在 `domain/ports.py` 增加 `VisionPort`；
2. 在 `adapters/` 实现 DeepStream/TensorRT Adapter；
3. 实现一个 `CapabilityProvider`，输出带置信度与来源的视觉 Artifact；
4. 由 Plugin 或组合根将 Provider 注册到 `CapabilityRegistry`；
5. 增加视觉事件、超时、资源限额和测试。

连续视频保持在 GPU/共享内存数据面，事件总线只传结构化结果和 `frame_ref`。人员检测、碰撞预警等实时安全视觉不得依赖 Agent Runner，必须进入独立 Safety Controller。

## 下一阶段

推荐按以下顺序替换演示实现：

1. 实现 vLLM Adapter 和结构化 Tool Calling；
2. 接入只读 CAN/J1939 遥测网关；
3. 实现带机型、序列号和手册版本过滤的本地 RAG；
4. 接入 Riva 流式 ASR/TTS 与打断处理；
5. 加密事件存储、知识包签名、离线同步和 OTA 回滚；
6. 独立设计并验证 Safety Command Gateway；
7. 最后增加视觉能力。
