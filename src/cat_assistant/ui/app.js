/* ============================================================
   MechDiagBot · 调试控制台 — 前端逻辑(零依赖原生 JS)
   复用后端 API 契约:/api/state /api/turn /api/config/* /api/remember
   ============================================================ */

let MACHINE_ID = null;
let SESSION_ID = '';
const OPERATOR_ID = 'debug-operator';

const state = {
  machine: null,
  events: [],
  messages: [],
  configuration: null,
  runtimePlugins: [],
  eventFilter: '',
  activeView: 'console',
  activeSettings: 'model',
  editingModelId: '',
  turnController: null,
  machines: [],
  sessions: [],
  faultCatalog: {},
  active: { machine: null, byMachine: {} },
  sessionMenuOpen: false,
};

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>'"]/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]
));
const csv = (value) => String(value || '').split(',').map((s) => s.trim()).filter(Boolean);
const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'short', timeStyle: 'medium',
});
const timeFormatter = new Intl.DateTimeFormat(undefined, { timeStyle: 'medium' });

const ARTIFACT_LABELS = {
  machine_snapshot: '设备快照', telemetry_summary: '遥测摘要', knowledge_evidence: '知识证据',
  diagnostic_assessment: '诊断评估', action_proposal: '行动建议',
  action_result: '执行结果', verification_report: '复核结果',
  indicator_report: '指示灯报告', audio_alarm_report: '声音报警',
};

/* ---------------- 网络 ---------------- */
async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
async function post(path, payload, options = {}) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

let toastTimer;
function toast(message, error = false) {
  const node = $('toast');
  node.textContent = message;
  node.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.className = 'toast'; }, 2800);
}

function setConnection(online) {
  const node = $('conn-status');
  node.className = online ? 'conn' : 'conn offline';
  node.innerHTML = `<i></i>${online ? '本地调试' : '连接断开'}`;
}

function setBusy(button, busy, loadingLabel = '处理中…') {
  if (!button) return;
  button.disabled = busy;
  const label = button.querySelector('.button-label');
  if (label) {
    if (busy) {
      button.dataset.idleLabel = label.textContent;
      label.textContent = loadingLabel;
    } else {
      label.textContent = button.dataset.idleLabel || label.textContent;
    }
  }
  button.setAttribute('aria-busy', String(busy));
}

function showFormError(id, message = '') {
  const node = $(id);
  if (!node) return;
  node.textContent = message;
  node.hidden = !message;
  if (message) node.focus();
}

function updateUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set('view', state.activeView);
  if (state.activeView === 'settings') url.searchParams.set('settings', state.activeSettings);
  else url.searchParams.delete('settings');
  if (state.activeView === 'audit' && state.eventFilter) url.searchParams.set('event', state.eventFilter);
  else url.searchParams.delete('event');
  if (url.href !== window.location.href) {
    window.history.pushState({ view: state.activeView, settings: state.activeSettings }, '', url);
  }
}

/* ---------------- 渲染:设备 ---------------- */
// 设备状态语义色:作业中=绿、重载作业=琥珀、怠速=中性、熄火=灰;发动机 运行中=绿 / 已停止=灰。
const STATE_TONE = { '作业中': 'ok', '重载作业': 'warn', '怠速': 'idle', '熄火': 'off' };
const stateTone = (s) => STATE_TONE[s] || 'neutral';
const engineTone = (running) => (running ? 'ok' : 'off');
// 指示灯语义:停机=灰(off)、运行有故障=红(fault)、运行正常=绿(ok)。
// 侧栏设备圆点与顶部状态条状态灯共用此函数,确保两处颜色与呼吸完全一致。
const dotTone = (m) => (!m.engine_running ? 'off' : ((m.fault_codes || []).length ? 'fault' : 'ok'));
function setTone(el, text, tone) {
  el.textContent = text;
  el.className = 'st-' + tone;
}

function renderStatusbar(machine) {
  $('q-model').textContent = machine.model;
  $('q-id').textContent = machine.machine_id;
  setTone($('q-state'), machine.operating_state, stateTone(machine.operating_state));
  setTone($('q-engine'), machine.engine_running ? '运行中' : '已停止', engineTone(machine.engine_running));
  $('q-fuel').textContent = `${Math.round(machine.fuel_percent)}%`;
  const faults = machine.fault_codes || [];
  $('q-fault').textContent = faults.length ? faults.join(' · ') : '无';
  document.querySelector('.sb-metric-fault').classList.toggle('has-fault', faults.length > 0);
  // 顶部头像上的状态灯与侧栏设备圆点同色同步:停机=灰、运行=绿/红并呼吸。
  $('q-avatar').className = 'sb-avatar ' + dotTone(machine);
}

function renderMachine(machine) {
  state.machine = machine;
  const faults = machine.fault_codes || [];
  const remaining = Math.max(0, machine.next_service_hours - machine.hour_meter);
  const fuel = Math.round(machine.fuel_percent);

  $('m-captured').textContent = timeFormatter.format(new Date(machine.captured_at));
  $('m-model').textContent = machine.model;
  $('m-serial').textContent = machine.serial_number;
  $('m-id').textContent = machine.machine_id;
  $('m-state').textContent = machine.operating_state;
  $('m-state-pill').className = 'state-pill ' + stateTone(machine.operating_state);
  $('m-fuel').textContent = `${fuel}%`;
  const bar = $('m-fuel-bar');
  bar.style.width = `${fuel}%`;
  bar.classList.toggle('low', fuel < 25);
  setTone($('m-engine'), machine.engine_running ? '运行中' : '已停止', engineTone(machine.engine_running));
  $('m-hours').textContent = Number(machine.hour_meter).toFixed(1);
  $('m-service').textContent = `${remaining.toFixed(1)}h`;
  $('m-fault-count').textContent = faults.length;

  $('fault-list').innerHTML = faults.length
    ? faults.map((code) => `<div class="fault-item"><span class="f-dot"></span><span class="f-code">${esc(code)}</span><span class="f-label">活动故障码</span></div>`).join('')
    : '<span class="muted">无活动故障码</span>';

  const facts = [
    ['机型', machine.model],
    ['序列号', machine.serial_number],
    ['设备 ID', machine.machine_id],
    ['工况', machine.operating_state, stateTone(machine.operating_state)],
    ['发动机', machine.engine_running ? '运行中' : '已停止', engineTone(machine.engine_running)],
    ['燃油', `${fuel}%`],
    ['累计小时', `${Number(machine.hour_meter).toFixed(1)} h`],
    ['距下次保养', `${remaining.toFixed(1)} h`],
    ['故障码', faults.join('、') || '无'],
  ];
  $('facts-list').innerHTML = facts
    .map(([k, v, tone]) => `<div class="fact-row"><span>${esc(k)}</span><b${tone ? ` class="st-${tone}"` : ''}>${esc(v)}</b></div>`).join('');
}

/* ---------------- 渲染:对话 ---------------- */
// 故障码释义来自后端 /api/state 的 fault_catalog;编译成一个整体正则用于在消息
// 文本里定位这些码。长码优先(先试 E456 再试 E45),避免短码抢先吃掉长码前缀。
let faultPattern = null;
function setFaultCatalog(catalog) {
  state.faultCatalog = (catalog && typeof catalog === 'object') ? catalog : {};
  const codes = Object.keys(state.faultCatalog).sort((a, b) => b.length - a.length);
  faultPattern = codes.length
    ? new RegExp(`\\b(${codes.map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'g')
    : null;
}
// 把已转义的消息 HTML 中出现的已知故障码替换为可悬停查看释义的链接。故障码只含
// 字母数字,转义不改变其文本,故可安全地在转义结果上替换,插入的释义也一并转义。
function linkifyFaults(escaped) {
  if (!faultPattern) return escaped;
  return escaped.replace(faultPattern, (code) => {
    const desc = state.faultCatalog[code];
    if (!desc) return code;
    return `<span class="fault-ref" tabindex="0" role="note" aria-label="故障码 ${esc(code)}：${esc(desc)}" data-desc="${esc(desc)}">${code}</span>`;
  });
}

const AVATAR_BOT = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2l2 8 8 2-8 2-2 8-2-8-8-2 8-2z"/></svg>';
const AVATAR_USER = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8.5" r="3.6"/><path d="M5.5 19.5c.8-3.7 3.4-5.6 6.5-5.6s5.7 1.9 6.5 5.6"/></svg>';

function renderConversation() {
  const box = $('conversation');
  if (!state.messages.length) {
    box.innerHTML = MACHINE_ID
      ? '<div class="empty-state"><span class="empty-glyph" aria-hidden="true"></span><b>有什么可以帮您？</b></div>'
      : '<div class="empty-state"><span class="empty-glyph" aria-hidden="true"></span><b>尚未接入设备</b><span>可直接进行通用问答；接入左侧设备后可查询实时状态、故障诊断与设备操作。</span></div>';
    return;
  }
  box.innerHTML = state.messages.map((m) => {
    const isUser = m.role === 'user';
    return `
    <div class="message ${m.role}${m.error ? ' error' : ''}${m.pending ? ' pending' : ''}">
      <div class="msg-avatar" aria-hidden="true">${isUser ? AVATAR_USER : AVATAR_BOT}</div>
      <div class="msg-main">
        <div class="message-body">${linkifyFaults(esc(m.text))}</div>
        ${m.approval ? '<div class="approval-actions"><button class="mini-button approval-yes" data-action="approve-turn">执行</button><button class="mini-button" data-action="reject-turn">取消</button></div>' : ''}
      </div>
    </div>`;
  }).join('');
  box.scrollTop = box.scrollHeight;
}

/* ---------------- 渲染:执行流水线 ---------------- */
function renderPipeline(metadata) {
  const meta = metadata || {};
  const steps = meta.steps || [];
  const caps = meta.capabilities || [];
  const artifacts = meta.artifacts || [];

  $('plan-steps').innerHTML = steps.length
    ? steps.map((s, i) => `<li><b>${esc(s)}</b><small>${esc(caps[i] || s)}</small></li>`).join('')
    : '<li class="muted">暂无阶段</li>';

  $('cap-chain').innerHTML = caps.length
    ? caps.map((c) => `<span class="cap-pill">${esc(c)}</span>`).join('')
    : '<span class="muted">—</span>';

  $('artifact-count').textContent = artifacts.length;
  $('artifact-list').innerHTML = artifacts.length
    ? artifacts.map((a) => `
        <div class="artifact-item">
          <div>
            <div class="a-type">${esc(ARTIFACT_LABELS[a.artifact_type] || a.artifact_type)}</div>
            <div class="a-src">${esc(a.source_capability)}</div>
          </div>
          <span class="a-conf">${Math.round((a.confidence ?? 1) * 100)}%</span>
        </div>`).join('')
    : '<span class="muted">暂无产物</span>';

  const sources = artifacts.filter((a) => /knowledge|evidence|manual|source/i.test(a.artifact_type));
  $('source-list').innerHTML = sources.length
    ? sources.map((a) => `<div class="source-item"><b>${esc(a.source_capability)}</b><span>置信度 ${Math.round((a.confidence ?? 1) * 100)}% · ${esc(ARTIFACT_LABELS[a.artifact_type] || a.artifact_type)}</span></div>`).join('')
    : '<span class="muted">暂无来源</span>';
}

/* ---------------- 渲染:能力目录 ---------------- */
function renderCapabilities(data) {
  const caps = data.capabilities || [];
  const tools = data.tools || [];
  const plugins = data.plugins || [];
  const mcp = data.mcp_servers || [];
  state.runtimePlugins = plugins;

  $('cap-count').textContent = caps.length;
  $('tool-count').textContent = tools.length;
  $('plugin-runtime-count').textContent = plugins.length;
  $('mcp-count').textContent = mcp.length;

  $('capability-list').innerHTML = caps.length ? caps.map((c) => {
    const tone = !c.enabled ? '' : c.requires_approval ? 'warn' : 'ok';
    const label = !c.enabled ? 'DISABLED' : c.requires_approval ? 'APPROVAL' : String(c.risk_level || 'low').toUpperCase();
    const flow = [(c.consumes || []).join(', ') || '∅', (c.produces || []).join(', ') || '∅'].join(' → ');
    return `<div class="tile"><div class="tile-icon c">C</div><div class="tile-copy"><b>${esc(c.name)}</b><span>v${esc(c.version)} · ${esc(c.kind)} · ${esc(flow)}</span></div><span class="tile-status ${tone}">${esc(label)}</span></div>`;
  }).join('') : '<div class="empty-card"><b>暂无能力</b></div>';

  $('tool-list').innerHTML = tools.length ? tools.map((t) =>
    `<div class="tile"><div class="tile-icon t">T</div><div class="tile-copy"><b>${esc(t.name)}</b><span>${esc(t.description || '')}</span></div><span class="tile-status ${t.read_only ? 'read' : 'warn'}">${t.read_only ? 'READ' : 'WRITE'}</span></div>`
  ).join('') : '<div class="empty-card"><b>暂无工具</b></div>';

  $('plugin-runtime-list').innerHTML = plugins.length ? plugins.map((p) =>
    `<div class="tile"><div class="tile-icon p">P</div><div class="tile-copy"><b>${esc(p.name)}</b><span>v${esc(p.version)} · ${esc(p.description || '内置插件')}</span></div><span class="tile-status ok">运行中</span></div>`
  ).join('') : '<div class="empty-card"><b>暂无插件</b></div>';

  $('mcp-runtime-list').innerHTML = mcp.length ? mcp.map((s) =>
    `<div class="tile"><div class="tile-icon m">M</div><div class="tile-copy"><b>${esc(s.name)}</b><span>${esc(s.transport)} · ${esc(s.endpoint || s.command || '')}</span></div><span class="tile-status warn">待连接</span></div>`
  ).join('') : '<div class="empty-card"><b>暂无 MCP</b></div>';
}

/* ---------------- 渲染:审计事件 ---------------- */
function renderEvents(events) {
  state.events = events || [];
  const types = [...new Set(state.events.map((e) => e.event_type))].sort();
  const filter = $('event-filter');
  const current = state.eventFilter;
  filter.innerHTML = '<option value="">全部</option>' +
    types.map((t) => `<option value="${esc(t)}"${t === current ? ' selected' : ''}>${esc(t)}</option>`).join('');

  const rows = state.events
    .filter((e) => !current || e.event_type === current)
    .slice().reverse();
  $('event-list').innerHTML = rows.length ? rows.map((e) => {
    const payload = JSON.stringify(e.payload || {}, null, 2);
    return `<details class="event-row">
      <summary><span class="event-time">${esc(dateFormatter.format(new Date(e.occurred_at)))}</span>
        <b class="event-type">${esc(e.event_type)}</b>
        <span class="event-payload">${esc(JSON.stringify(e.payload || {}))}</span></summary>
      <pre class="event-details">${esc(payload)}</pre>
    </details>`;
  }).join('') : '<div class="empty-state"><b>暂无事件</b><span>运行一轮诊断后会显示追加日志。</span></div>';
}

/* ---------------- 渲染:运行时配置 ---------------- */
function configRow(icon, name, detail, status, tone, actions) {
  return `<div class="config-row"><div class="config-icon">${esc(icon)}</div><div class="config-copy"><b>${esc(name)}</b><span>${esc(detail || '—')}</span></div><span class="chip ${tone}">${esc(status)}</span>${actions ? `<div class="config-actions">${actions}</div>` : ''}</div>`;
}

/* 模型:每个 profile 自带连接/推理/超时参数,切模型等于把这一整套切过去。
   列表选中谁,下面的编辑器就编辑谁;active_model_id 才是运行时真正在用的那个。*/
const NEW_MODEL_ID = '__new__';
const PROVIDER_LABELS = {
  demo: 'Demo', vllm: 'vLLM', ollama: 'Ollama',
  openai_compatible: 'OpenAI Compatible', axllm: 'AXLLM',
};
const MODEL_DRAFT = {
  id: '', name: '', provider: 'vllm', model: '', base_url: '', api_key: '',
  temperature: 0.1, max_tokens: 1024, timeout_seconds: 30, max_steps: 4,
  planning_mode: 'auto', disable_thinking: false,
  model_call_timeout_seconds: 60, tool_call_timeout_seconds: 30, turn_timeout_seconds: 120,
};

const modelProfiles = () => state.configuration?.models || [];
const providerLabel = (provider) => PROVIDER_LABELS[provider] || provider;

function editingProfile() {
  if (state.editingModelId === NEW_MODEL_ID) return { ...MODEL_DRAFT };
  return modelProfiles().find((profile) => profile.id === state.editingModelId) || null;
}

function renderModelProfiles() {
  const activeId = state.configuration?.active_model_id;
  const card = (profile, { draft = false } = {}) => {
    const selected = draft || profile.id === state.editingModelId;
    return `<button type="button" class="profile-card${selected ? ' active' : ''}"
      data-action="edit-model" data-id="${esc(draft ? NEW_MODEL_ID : profile.id)}" aria-pressed="${selected}">
      <span class="profile-head"><b>${esc(profile.name || profile.model || '新建模型')}</b>${
        draft ? '' : (profile.id === activeId ? '<span class="chip chip-success">使用中</span>' : '')
      }</span>
      <span class="profile-meta">${draft ? '尚未保存' : `${esc(providerLabel(profile.provider))} · ${esc(profile.model)}`}</span>
    </button>`;
  };
  const cards = modelProfiles().map((profile) => card(profile)).join('');
  $('model-list').innerHTML = cards
    + (state.editingModelId === NEW_MODEL_ID ? card(MODEL_DRAFT, { draft: true }) : '');
}

function renderModelHead() {
  const profile = editingProfile();
  const isNew = state.editingModelId === NEW_MODEL_ID;
  const isActive = !isNew && profile?.id === state.configuration?.active_model_id;
  const applied = state.configuration?.runtime?.model_config_applied;
  $('model-title').textContent = isNew ? '新建模型' : (profile?.name || profile?.model || '—');
  $('model-subtitle').textContent = isNew ? '保存后加入列表' : (profile?.id || '—');
  const chip = $('model-status-chip');
  const status = isNew ? ['未保存', 'chip-neutral']
    : !isActive ? ['未启用', 'chip-neutral']
    : applied ? ['运行中', 'chip-success'] : ['配置待应用', 'chip-warning'];
  chip.textContent = status[0];
  chip.className = `chip ${status[1]}`;
  // 已在用的 profile 不提供「设为当前」;最后一个 profile 不能删,否则运行时无模型可启动。
  $('activate-model-btn').hidden = isNew || isActive;
  $('delete-model-btn').hidden = isNew || modelProfiles().length <= 1;
}

function renderModelEditor({ force = false } = {}) {
  renderModelHead();
  const profile = editingProfile();
  if (!profile) return;
  // 后台刷新(轮询/一轮诊断结束)不覆盖正在编辑的未保存内容。
  if (!force && $('model-form').dataset.dirty === 'true') return;
  $('model-label').value = profile.name === profile.model ? '' : (profile.name || '');
  $('model-provider').value = profile.provider;
  $('model-name').value = profile.model;
  $('model-base-url').value = profile.base_url;
  $('model-api-key').value = profile.api_key;
  $('model-temperature').value = profile.temperature;
  $('model-max-tokens').value = profile.max_tokens;
  $('model-timeout').value = profile.timeout_seconds;
  $('model-max-steps').value = profile.max_steps;
  $('model-call-timeout').value = profile.model_call_timeout_seconds;
  $('tool-call-timeout').value = profile.tool_call_timeout_seconds;
  $('turn-timeout').value = profile.turn_timeout_seconds;
  $('model-planning-mode').value = profile.planning_mode || 'auto';
  $('model-disable-thinking').checked = Boolean(profile.disable_thinking);
  $('model-form').dataset.dirty = 'false';
}

function selectModel(itemId, { force = false } = {}) {
  if (itemId === state.editingModelId) return;
  if (!force && !confirmUnsavedModelChanges()) return;
  state.editingModelId = itemId;
  $('model-save-msg').textContent = '';
  showFormError('model-form-error');
  renderModelProfiles();
  renderModelEditor({ force: true });
}

function renderConfiguration(configuration) {
  state.configuration = configuration;
  const ids = modelProfiles().map((profile) => profile.id);
  if (state.editingModelId !== NEW_MODEL_ID && !ids.includes(state.editingModelId)) {
    state.editingModelId = configuration.active_model_id;
  }
  renderModelProfiles();
  renderModelEditor();

  const applied = configuration.runtime.model_config_applied;
  $('runtime-adapter').textContent = configuration.runtime.model_adapter;
  $('runtime-applied').textContent = applied ? '已应用' : '待适配器接入';
  $('runtime-planning').textContent = configuration.model.planning_mode || 'auto';

  $('mcp-config-list').innerHTML = configuration.mcp_servers.map((s) => configRow(
    'MCP', s.name,
    s.transport === 'stdio' ? s.command : s.endpoint,
    s.enabled ? '待启用' : '已配置', s.enabled ? 'chip-warning' : 'chip-neutral',
    `<button class="mini-button" data-action="edit-mcp" data-id="${esc(s.id)}">管理</button><button class="mini-button danger" data-action="delete-mcp" data-id="${esc(s.id)}">移除</button>`
  )).join('');

  $('builtin-plugin-list').innerHTML = state.runtimePlugins.map((p) => configRow(
    'CORE', p.name, `v${p.version} · ${p.description || '内置插件'}`, '运行中', 'chip-success', ''
  )).join('');

  $('plugin-config-list').innerHTML = configuration.plugins.map((p) => configRow(
    'PLG', p.name, `${p.reference}${p.version ? ` · ${p.version}` : ''}`,
    p.enabled ? '待启用' : '已配置', p.enabled ? 'chip-warning' : 'chip-neutral',
    `<button class="mini-button" data-action="edit-plugin" data-id="${esc(p.id)}">管理</button><button class="mini-button danger" data-action="delete-plugin" data-id="${esc(p.id)}">移除</button>`
  )).join('');
}

/* ---------------- 设备接入 ---------------- */
// 启动时保持"未接入设备"：智能体不预绑定任何设备，点击左侧设备后才接入其状态。
// 这样与 OBD 诊断仪一致——物理/手动选择设备之前，控制台只提供配置与设备列表。
function renderUnboundStatusbar() {
  $('q-model').textContent = '未接入设备';
  $('q-id').textContent = '请选择左侧设备';
  setTone($('q-state'), '—', 'neutral');
  setTone($('q-engine'), '—', 'off');
  $('q-fuel').textContent = '—';
  $('q-fault').textContent = '—';
  document.querySelector('.sb-metric-fault')?.classList.remove('has-fault');
  $('q-avatar').className = 'sb-avatar off';
}

function setComposerEnabled(enabled) {
  const input = $('turn-input');
  const submit = $('turn-submit');
  const bound = Boolean(MACHINE_ID);
  if (input) {
    input.disabled = !enabled;
    input.placeholder = enabled
      ? (bound ? COMPOSER_PLACEHOLDER : GENERAL_PLACEHOLDER)
      : '暂时无法发送消息…';
  }
  if (submit) submit.disabled = !enabled;
  input?.closest('.composer-box')?.classList.toggle('is-disabled', !enabled);
  if (!enabled && voice.active) stopVoiceSession();
  if (!voice.active) {
    setVoiceHint(enabled ? (bound ? '' : GENERAL_HINT) : '暂时无法发送消息');
  }
  const mic = $('voice-btn');
  if (mic) mic.hidden = !(enabled && voice.available);
}

/* ---------------- 加载 ---------------- */
async function loadState() {
  if (!MACHINE_ID) {
    // 未接入设备时仍可进行通用问答；只加载无设备上下文和通用会话历史。
    try {
      const data = await request(`/api/state?machine_id=&session_id=${encodeURIComponent(SESSION_ID)}&operator_id=${encodeURIComponent(OPERATOR_ID)}`);
      setFaultCatalog(data.fault_catalog);
      renderUnboundStatusbar();
      renderCapabilities(data);
      renderEvents(data.events);
      renderConfiguration(data.configuration);
      if (Array.isArray(data.turns) && !state.messages.length) {
        state.messages = data.turns.flatMap((turn) => [
          { role: 'user', text: turn.user_text },
          { role: 'assistant', text: turn.assistant_text },
        ]);
      }
      renderConversation();
      renderSessions();
      setComposerEnabled(true);
      setConnection(true);
    } catch (error) {
      setConnection(false);
      throw error;
    }
    return;
  }
  try {
    const data = await request(`/api/state?machine_id=${encodeURIComponent(MACHINE_ID)}&session_id=${encodeURIComponent(SESSION_ID)}&operator_id=${encodeURIComponent(OPERATOR_ID)}`);
    setFaultCatalog(data.fault_catalog);
    renderStatusbar(data.machine);
    renderMachine(data.machine);
    renderCapabilities(data);
    renderEvents(data.events);
    renderConfiguration(data.configuration);
    if (Array.isArray(data.turns) && !state.messages.length) {
      state.messages = data.turns.flatMap((turn) => [
        { role: 'user', text: turn.user_text },
        { role: 'assistant', text: turn.assistant_text },
      ]);
      renderConversation();
    }
    setComposerEnabled(true);
    setConnection(true);
  } catch (error) {
    setConnection(false);
    throw error;
  }
}

/* ---------------- 设备切换 + 会话管理 ---------------- */
const LS_SESSIONS = 'catedge.sessions.v1';
const LS_ACTIVE = 'catedge.active.v1';

function readStore(key, fallback) {
  try { const raw = JSON.parse(localStorage.getItem(key)); return raw ?? fallback; }
  catch { return fallback; }
}
function persistSessions() { try { localStorage.setItem(LS_SESSIONS, JSON.stringify(state.sessions)); } catch {} }
function persistActive() { try { localStorage.setItem(LS_ACTIVE, JSON.stringify(state.active)); } catch {} }

function newSessionId() {
  return (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : `s-${Date.now()}-${Math.round(Math.random() * 1e6)}`;
}
function sessionsForMachine(machineId) {
  return state.sessions.filter((s) => s.machineId === machineId).sort((a, b) => b.updatedAt - a.updatedAt);
}
function currentSession() {
  return state.sessions.find((s) => s.id === SESSION_ID) || null;
}
function createSession(machineId, { activate = true } = {}) {
  const now = Date.now();
  const session = { id: newSessionId(), title: '新会话', machineId, createdAt: now, updatedAt: now };
  state.sessions.push(session);
  persistSessions();
  if (activate) { SESSION_ID = session.id; state.active.byMachine[machineId] = session.id; persistActive(); }
  return session;
}
function ensureSessionForMachine(machineId) {
  const wanted = state.active.byMachine[machineId];
  const existing = state.sessions.find((s) => s.id === wanted && s.machineId === machineId) || sessionsForMachine(machineId)[0];
  if (existing) { SESSION_ID = existing.id; state.active.byMachine[machineId] = existing.id; persistActive(); return existing; }
  return createSession(machineId);
}
function touchSession(userText) {
  const session = currentSession();
  if (!session) return;
  session.updatedAt = Date.now();
  if (!session.title || session.title === '新会话') {
    const trimmed = userText.trim();
    session.title = trimmed.slice(0, 18) + (trimmed.length > 18 ? '…' : '');
  }
  persistSessions();
  renderSessions();
}

function renderMachineSwitcher() {
  const box = $('machine-switcher');
  $('machine-total').textContent = state.machines.length;
  if (!state.machines.length) { box.innerHTML = '<div class="switcher-empty">无可用设备</div>'; return; }
  box.innerHTML = state.machines.map((m) => {
    const faults = m.fault_codes || [];
    // 停机=灰不呼吸;运行时红(有故障)/绿(正常)并做呼吸灯,与顶部状态灯同色同步。
    const tone = dotTone(m);
    const active = m.machine_id === MACHINE_ID;
    return `<button type="button" class="machine-item${active ? ' active' : ''}" role="option" aria-selected="${active}" data-machine="${esc(m.machine_id)}">
      <span class="mi-dot ${tone}" aria-hidden="true"></span>
      <span class="mi-copy"><b>${esc(m.model)}</b><span>${esc(m.machine_id)}</span></span>
      ${faults.length ? `<span class="mi-badge" title="活动故障">${faults.length}</span>` : ''}
    </button>`;
  }).join('');
}

function renderSessions() {
  const machineKey = MACHINE_ID || '';
  const list = sessionsForMachine(machineKey);
  const cur = currentSession();
  $('session-name').textContent = cur ? cur.title : (MACHINE_ID ? '新会话' : '通用对话');
  const label = $('session-label');
  if (label) label.textContent = SESSION_ID.slice(0, 8);
  const newButton = $('new-session-btn');
  if (newButton) newButton.disabled = false;
  const deleteButton = $('delete-session-btn');
  if (deleteButton) deleteButton.disabled = !cur;
  $('session-toggle').disabled = false;
  $('session-menu').innerHTML = list.length ? list.map((s) => `
    <div class="session-item${s.id === SESSION_ID ? ' active' : ''}" role="option" aria-selected="${s.id === SESSION_ID}" data-session="${esc(s.id)}">
      <span class="si-copy"><b>${esc(s.title)}</b><span>${esc(dateFormatter.format(new Date(s.updatedAt)))}</span></span>
      <button type="button" class="si-del" data-del-session="${esc(s.id)}" aria-label="删除会话">×</button>
    </div>`).join('') : `<div class="switcher-empty">${MACHINE_ID ? '该设备暂无会话' : '暂无通用会话'}</div>`;
}

function openSessionMenu() { $('session-menu').hidden = false; $('session-toggle').setAttribute('aria-expanded', 'true'); state.sessionMenuOpen = true; }
function closeSessionMenu() { $('session-menu').hidden = true; $('session-toggle').setAttribute('aria-expanded', 'false'); state.sessionMenuOpen = false; }

async function reloadConsole() {
  state.messages = [];
  renderConversation();
  renderPipeline({});
  renderSessions();
  try { await loadState(); } catch (error) { toast(error.message, true); }
}

async function selectMachine(machineId) {
  if (!machineId) return;
  const wasBound = Boolean(MACHINE_ID);
  const sameMachine = machineId === MACHINE_ID;
  MACHINE_ID = machineId;
  state.active.machine = machineId;
  persistActive();
  ensureSessionForMachine(machineId);
  renderMachineSwitcher();
  setComposerEnabled(true);
  activateView('console');
  if (sameMachine) {
    $('turn-input')?.focus();
    return;
  }
  await reloadConsole();
  const machine = state.machines.find((m) => m.machine_id === machineId);
  toast(`${wasBound ? '已切换到' : '已接入'} ${machine ? machine.model : machineId}`);
}

async function selectGeneralAgent() {
  const wasBound = Boolean(MACHINE_ID);
  stopVoicePlayback();
  MACHINE_ID = null;
  state.active.machine = null;
  persistActive();
  ensureSessionForMachine('');
  renderMachineSwitcher();
  setComposerEnabled(true);
  activateView('console');
  if (wasBound) {
    await reloadConsole();
    toast('已切换到通用助手');
  }
  $('turn-input')?.focus();
}

async function selectSession(id) {
  if (id === SESSION_ID) { closeSessionMenu(); return; }
  if (!state.sessions.some((s) => s.id === id)) { closeSessionMenu(); return; }
  SESSION_ID = id;
  state.active.byMachine[MACHINE_ID || ''] = id;
  persistActive();
  closeSessionMenu();
  await reloadConsole();
}

async function newSession() {
  createSession(MACHINE_ID || '');
  closeSessionMenu();
  renderMachineSwitcher();
  await reloadConsole();
  $('turn-input')?.focus();
}

async function deleteSession(id) {
  const index = state.sessions.findIndex((s) => s.id === id);
  if (index < 0) return;
  // Delete the backend history/summary first so the UI never diverges from the
  // store. A frontend-only session that never ran a turn simply removes 0 rows.
  try {
    await post('/api/session/delete', { session_id: id });
  } catch (error) {
    toast(`后端会话历史删除失败：${error.message}`, true);
    return;
  }
  const wasCurrent = id === SESSION_ID;
  state.sessions.splice(index, 1);
  persistSessions();
  if (wasCurrent) {
    ensureSessionForMachine(MACHINE_ID || '');
    await reloadConsole();
  } else {
    renderSessions();
  }
  toast('会话及后端历史已删除。');
}

let fleetSignature = '';
function fleetSig(machines) {
  return machines
    .map((m) => `${m.machine_id}:${m.engine_running ? 1 : 0}:${m.operating_state || ''}:${(m.fault_codes || []).join(',')}`)
    .join('|');
}

// 把当前设备的最新快照并入机队列表并即时刷新侧栏指示灯。控制指令(停机/启动
// 等)执行后调用,让侧栏圆点与顶部状态灯无需等待轮询即随之变色;同时对齐
// fleetSignature,避免下一次 pollFleet 因"看似有变化"而重绘、打断呼吸灯动画。
function syncActiveMachine(machine) {
  if (!machine || !machine.machine_id) return;
  const index = state.machines.findIndex((m) => m.machine_id === machine.machine_id);
  if (index >= 0) state.machines[index] = { ...state.machines[index], ...machine };
  else state.machines.push(machine);
  fleetSignature = fleetSig(state.machines);
  renderMachineSwitcher();
}

async function loadMachines() {
  try {
    const data = await request('/api/machines');
    state.machines = Array.isArray(data.machines) ? data.machines : [];
  } catch { state.machines = []; }
  fleetSignature = fleetSig(state.machines);
  renderMachineSwitcher();
}

// 机队每 10 分钟轮换一次活动故障;轻量轮询让指示灯随之刷新,且仅在实际
// 发生变化时才重绘设备列表,避免无谓地打断"呼吸灯"动画。
async function pollFleet() {
  let machines;
  try {
    const data = await request('/api/machines');
    machines = Array.isArray(data.machines) ? data.machines : [];
  } catch { return; } // 瞬时错误:保留上一次的机队状态
  const sig = fleetSig(machines);
  if (sig === fleetSignature) return;
  fleetSignature = sig;
  state.machines = machines;
  renderMachineSwitcher();
  // 顺带用机队快照刷新当前设备的状态栏与详情,避免与指示灯不一致;不走
  // loadState,以免覆盖用户在配置页尚未保存的输入。
  const activeMachine = machines.find((m) => m.machine_id === MACHINE_ID);
  if (activeMachine) { renderStatusbar(activeMachine); renderMachine(activeMachine); }
}

async function bootstrapConsole() {
  state.sessions = readStore(LS_SESSIONS, []);
  if (!Array.isArray(state.sessions)) state.sessions = [];
  const active = readStore(LS_ACTIVE, { machine: null, byMachine: {} });
  state.active = (active && typeof active === 'object' && active.byMachine) ? active : { machine: null, byMachine: {} };

  await loadMachines();
  // 不恢复上次的设备绑定：每次打开都从未接入状态开始，由用户点击设备后再接入。
  // 通用会话独立于设备，未接入时也能直接问答。
  MACHINE_ID = null;
  state.active.machine = null;
  ensureSessionForMachine('');
  persistActive();
  renderMachineSwitcher();
  await loadState();
  // 保持机队指示灯"活"着:每分钟对齐一次后端每 10 分钟的故障轮换。
  setInterval(pollFleet, 60000);
}

/* 设备切换器 / 会话下拉的事件委托 */
$('machine-switcher').addEventListener('click', (event) => {
  const button = event.target.closest('[data-machine]');
  if (!button) return;
  if (state.activeView === 'settings' && state.activeSettings === 'model' && !confirmUnsavedModelChanges()) return;
  selectMachine(button.dataset.machine);
});
$('session-toggle').addEventListener('click', () => (state.sessionMenuOpen ? closeSessionMenu() : openSessionMenu()));
$('new-session-btn').addEventListener('click', () => newSession());
$('delete-session-btn').addEventListener('click', () => {
  if (!currentSession()) { toast('当前没有可删除的会话。'); return; }
  deleteSession(SESSION_ID);
});
$('session-menu').addEventListener('click', (event) => {
  const del = event.target.closest('[data-del-session]');
  if (del) { event.stopPropagation(); deleteSession(del.dataset.delSession); return; }
  const item = event.target.closest('[data-session]');
  if (item) selectSession(item.dataset.session);
});
document.addEventListener('click', (event) => {
  if (state.sessionMenuOpen && !event.target.closest('.session-picker')) closeSessionMenu();
});

/* ---------------- 对话框 ---------------- */
function openMcp(server = null) {
  $('mcp-form').reset();
  $('mcp-id').value = server?.id || '';
  $('mcp-name').value = server?.name || '';
  $('mcp-transport').value = server?.transport || 'streamable_http';
  $('mcp-endpoint').value = server?.endpoint || '';
  $('mcp-command').value = server?.command || '';
  $('mcp-arguments').value = (server?.arguments || []).join(', ');
  $('mcp-env-keys').value = (server?.env_keys || []).join(', ');
  $('mcp-tools').value = (server?.tool_allowlist || ['*']).join(', ');
  $('mcp-enabled').checked = Boolean(server?.enabled);
  $('mcp-dialog-title').textContent = server ? '管理 Server' : '添加 Server';
  $('mcp-dialog').showModal();
}
function openPlugin(plugin = null) {
  $('plugin-form').reset();
  $('plugin-id').value = plugin?.id || '';
  $('plugin-name').value = plugin?.name || '';
  $('plugin-version').value = plugin?.version || '';
  $('plugin-reference').value = plugin?.reference || '';
  $('plugin-config').value = JSON.stringify(plugin?.config || {}, null, 2);
  $('plugin-enabled').checked = Boolean(plugin?.enabled);
  $('plugin-dialog-title').textContent = plugin ? '管理插件' : '添加插件';
  $('plugin-dialog').showModal();
}

/* ---------------- 事件:任务编排 ---------------- */
// 发送一轮文本任务：语音识别的结果也复用这条路径。
async function sendTurn(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return null;
  stopVoicePlayback();
  const button = $('turn-submit');
  const cancel = $('turn-cancel');
  const input = $('turn-input');
  showFormError('turn-form-error');
  state.turnController?.abort();
  state.turnController = new AbortController();
  setBusy(button, true, '执行中…');
  cancel.hidden = false;
  state.messages.push({ role: 'user', text: trimmed });
  state.messages.push({ role: 'assistant', text: '思考中…', pending: true });
  renderConversation();
  if (input.value.trim() === trimmed) {
    input.value = '';
    autosizeComposer();
  }
  try {
    const data = await post('/api/turn', {
      text: trimmed, session_id: SESSION_ID, machine_id: MACHINE_ID || '', operator_id: OPERATOR_ID,
    }, { signal: state.turnController.signal });
    state.messages = state.messages.filter((message) => !message.pending);
    const meta = data.response.metadata || {};
    state.messages.push({ role: 'assistant', text: data.response.text, approval: Boolean(data.response.requires_confirmation) });
    renderConversation();
    renderPipeline(meta);
    touchSession(trimmed);
    if (data.state) {
      renderEvents(data.state.events);
      renderCapabilities(data.state);
      if (data.state.machine) {
        renderStatusbar(data.state.machine);
        renderMachine(data.state.machine);
        // 控制指令(停机/启动等)会改变设备状态:立即把新快照同步到侧栏设备
        // 列表,让指示灯即时变色,不必等下一次机队轮询。
        syncActiveMachine(data.state.machine);
      } else {
        // 通用会话没有设备快照,保持未接入状态条,不伪造遥测。
        renderUnboundStatusbar();
      }
    }
    return data;
  } catch (error) {
    state.messages = state.messages.filter((message) => !message.pending);
    const message = error.name === 'AbortError' ? '已停止等待本轮响应。' : `调试请求失败：${error.message}`;
    state.messages.push({ role: 'assistant', text: message, error: true });
    showFormError('turn-form-error', message);
    renderConversation();
    return null;
  } finally {
    state.turnController = null;
    setBusy(button, false);
    cancel.hidden = true;
  }
}

$('turn-form').addEventListener('submit', (event) => {
  event.preventDefault();
  sendTurn($('turn-input').value);
});

$('turn-cancel').addEventListener('click', () => state.turnController?.abort());
$('turn-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('turn-form').requestSubmit();
  }
});

// 输入框随内容增高（至多 168px），有内容时点亮发送键。
function autosizeComposer() {
  const input = $('turn-input');
  if (!input) return;
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 168)}px`;
  input.closest('.composer-box')?.classList.toggle('has-text', Boolean(input.value.trim()));
}
$('turn-input').addEventListener('input', autosizeComposer);

/* ---------------- 事件:网页语音(连续对话) ---------------- */
const COMPOSER_HINT = 'Enter 发送 · Shift + Enter 换行 · 点击麦克风开启语音对话';
const COMPOSER_PLACEHOLDER = '输入查询或故障诊断…';
const GENERAL_HINT = '通用问答 · 选择左侧设备后可查询实时状态';
const GENERAL_PLACEHOLDER = '输入通用问题…';
const WEB_SAMPLE_RATE = 16000;
const VOICE_CALIBRATION_MS = 300;
const VOICE_PRE_ROLL_MS = 500;
const VOICE_MIN_SPEECH_MS = 250;
const VOICE_SILENCE_MS = 1200;
const VOICE_MAX_UTTERANCE_MS = 15000;
const VOICE_GATE_FLOOR = 0.0015;
const VOICE_GATE_CEILING = 0.01;
const VOICE_NOISE_RATIO = 1.5;
const VOICE_NOISE_CAP = 0.005;
const VOICE_NO_SPEECH_REMINDER_MS = 6000;

const voice = {
  available: false,
  active: false,
  mode: 'idle',
  session: 0,
  stream: null,
  context: null,
  nodes: [],
  vad: null,
  noiseTimer: null,
  player: null,
  playback: null,
  sources: [],
  nextStart: 0,
  pcmRemainder: new Uint8Array(0),
  streamAbort: null,
};

function ensurePlaybackContext() {
  const AudioCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtor) return null;
  if (!voice.playback) voice.playback = new AudioCtor();
  if (voice.playback.state === 'suspended') {
    voice.playback.resume().catch(() => {});
  }
  return voice.playback;
}

function setVoiceHint(text) {
  const hint = $('composer-hint');
  if (hint) hint.textContent = text || COMPOSER_HINT;
}

function updateVoiceButton() {
  const button = $('voice-btn');
  if (!button) return;
  button.classList.toggle('recording', voice.active);
  button.classList.toggle('busy', voice.active && voice.mode !== 'listening');
  button.title = voice.active ? '结束语音对话' : '开启语音对话';
  button.setAttribute('aria-label', button.title);
  button.setAttribute('aria-pressed', String(voice.active));
}

function voiceListeningHint(first) {
  return first
    ? '语音对话已开启 · 直接说话，停顿后自动发送（再点麦克风结束）'
    : '请继续说话……（再点麦克风结束）';
}

function armNoSpeechReminder() {
  clearNoSpeechReminder();
  voice.noiseTimer = setTimeout(() => {
    if (voice.active && voice.mode === 'listening' && voice.vad && !voice.vad.active) {
      setVoiceHint('还没有听到声音 · 请检查麦克风权限或输入音量（再点麦克风结束）');
    }
  }, VOICE_NO_SPEECH_REMINDER_MS);
}

function clearNoSpeechReminder() {
  if (!voice.noiseTimer) return;
  clearTimeout(voice.noiseTimer);
  voice.noiseTimer = null;
}

function voiceGate(noise) {
  return Math.min(
    VOICE_GATE_CEILING,
    Math.max(VOICE_GATE_FLOOR, noise * VOICE_NOISE_RATIO),
  );
}

function mergeFloat32(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

// 浏览器采集采样率不一定是 16 kHz；线性插值降到 16 kHz，减少上传体积。
function downsampleFloat32(chunks, inputRate, targetRate = WEB_SAMPLE_RATE) {
  const input = mergeFloat32(chunks);
  if (!input.length || inputRate === targetRate) return input;
  const length = Math.floor((input.length * targetRate) / inputRate);
  if (length <= 0) return new Float32Array(0);
  const out = new Float32Array(length);
  const step = inputRate / targetRate;
  for (let i = 0; i < length; i += 1) {
    const position = i * step;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    out[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return out;
}

function float32ToS16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return out;
}

function voiceRms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function newVoiceVad() {
  return {
    calibratedMs: 0,
    noise: 0,
    active: false,
    pending: [],
    pendingMs: 0,
    chunks: [],
    voicedMs: 0,
    silenceMs: 0,
    speechRunMs: 0,
  };
}

// 点击一次进入连续对话：本地能量 VAD 自动判断停顿，无需一直按住按钮。
function handleVoiceFrame(samples, inputRate) {
  const vad = voice.vad;
  if (!vad || !samples.length) return;
  const frameMs = (samples.length / inputRate) * 1000;
  const level = voiceRms(samples);

  if (!vad.active) {
    // 保留起始前滚，避免触发瞬间吞掉句首。
    vad.pending.push({ samples, level, frameMs });
    vad.pendingMs += frameMs;
    while (vad.pending.length > 1 && vad.pendingMs > VOICE_PRE_ROLL_MS) {
      vad.pendingMs -= vad.pending.shift().frameMs;
    }
    if (vad.calibratedMs < VOICE_CALIBRATION_MS) {
      vad.calibratedMs += frameMs;
      // 底噪用前几帧的低值估计并设上限：用户点完立刻开口时，
      // 不能把句首语音当成底噪把门限抬到听不见。
      const sample = Math.min(level, VOICE_NOISE_CAP);
      vad.noise = vad.noise > 0 ? Math.min(vad.noise, sample) : sample;
    }
    if (!vad.noise) vad.noise = Math.min(level, VOICE_NOISE_CAP);
    const gate = voiceGate(vad.noise);
    if (level < gate) {
      if (vad.calibratedMs >= VOICE_CALIBRATION_MS) {
        const alpha = level > vad.noise ? 0.05 : 0.1;
        vad.noise = alpha * level + (1 - alpha) * vad.noise;
      }
      vad.speechRunMs = 0;
      return;
    }
    vad.speechRunMs += frameMs;
    if (vad.speechRunMs < 2 * frameMs) return;
    vad.active = true;
    vad.chunks = vad.pending.map((item) => item.samples);
    vad.voicedMs = vad.speechRunMs;
    vad.silenceMs = 0;
    vad.pending = [];
    vad.pendingMs = 0;
    clearNoSpeechReminder();
    setVoiceHint('正在聆听……停顿后自动发送');
    return;
  }

  vad.chunks.push(samples);
  const gate = voiceGate(vad.noise);
  if (level >= gate) {
    vad.voicedMs += frameMs;
    vad.silenceMs = 0;
  } else {
    vad.silenceMs += frameMs;
  }
  const durationMs = vad.chunks.length * frameMs;
  if (
    (vad.silenceMs >= VOICE_SILENCE_MS && vad.voicedMs >= VOICE_MIN_SPEECH_MS)
    || durationMs >= VOICE_MAX_UTTERANCE_MS
  ) {
    void finishVoiceUtterance();
  }
}

function resumeVoiceListening() {
  if (!voice.active || !voice.context) return;
  voice.mode = 'listening';
  voice.vad = newVoiceVad();
  updateVoiceButton();
  setVoiceHint(voiceListeningHint(false));
  armNoSpeechReminder();
}

async function waitVoicePlaybackDone() {
  while (voice.active) {
    if (voice.player) {
      if (voice.player.paused || voice.player.ended) return;
    } else {
      const context = voice.playback;
      if (!context || (!voice.sources.length && voice.nextStart <= context.currentTime + 0.05)) {
        return;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function finishVoiceUtterance() {
  const vad = voice.vad;
  if (!vad || !vad.active) return;
  vad.active = false;
  clearNoSpeechReminder();
  const chunks = vad.chunks;
  const voicedMs = vad.voicedMs;
  const inputRate = voice.context?.sampleRate || WEB_SAMPLE_RATE;
  voice.vad = newVoiceVad();

  if (voicedMs < VOICE_MIN_SPEECH_MS || !chunks.length) {
    setVoiceHint(voiceListeningHint(false));
    armNoSpeechReminder();
    return;
  }

  const token = voice.session;
  voice.mode = 'busy';
  updateVoiceButton();
  setVoiceHint('正在识别……');
  try {
    const samples = downsampleFloat32(chunks, inputRate);
    if (samples.length < WEB_SAMPLE_RATE * 0.2) {
      throw new Error('录音太短，请再说一次。');
    }
    const pcm = float32ToS16(samples);
    const response = await fetch(`/api/voice/transcribe?sample_rate=${WEB_SAMPLE_RATE}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: pcm,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `语音识别失败 (${response.status})`);
    if (token !== voice.session || !voice.active) return;
    const text = String(data.text || '').trim();
    if (!text) throw new Error('没有识别到语音，请再说一次。');
    const composer = $('turn-input');
    composer.value = text;
    autosizeComposer();
    toast(`识别：${text}`);
    setVoiceHint('正在生成回复……');
    const turn = await sendTurn(text);
    if (token !== voice.session || !voice.active) return;
    const reply = turn?.response?.text;
    if (reply) {
      setVoiceHint('正在朗读……');
      await speakReply(reply);
      await waitVoicePlaybackDone();
    }
  } catch (error) {
    if (token !== voice.session || !voice.active) return;
    toast(error.message || '语音交互失败。', true);
  } finally {
    // 朗读结束后自动回到聆听，形成连续对话；用户再点麦克风才结束。
    if (token === voice.session && voice.active) resumeVoiceListening();
  }
}

async function releaseVoiceCapture() {
  for (const node of voice.nodes) {
    try { node.disconnect(); } catch { /* 节点已断开 */ }
  }
  voice.nodes = [];
  // 先同步停掉麦克风并清空引用，再等音频上下文关闭；点击结束时
  // 立即释放设备，而不是等 close() 完成。
  const stream = voice.stream;
  voice.stream = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  const context = voice.context;
  voice.context = null;
  if (context) {
    try { await context.close(); } catch { /* 忽略关闭失败 */ }
  }
}

function createCaptureContext() {
  const AudioCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtor) throw new Error('浏览器不支持 Web Audio');
  try {
    return new AudioCtor({ sampleRate: WEB_SAMPLE_RATE });
  } catch {
    return new AudioCtor();
  }
}

async function startVoiceSession() {
  if (!voice.available || voice.active) return;
  const token = ++voice.session;
  voice.active = true;
  voice.mode = 'listening';
  voice.vad = newVoiceVad();
  stopVoicePlayback();
  // 借这次用户手势创建并解锁播放上下文；合成在几秒后才开始，届时
  // 浏览器不会再以“缺少用户交互”为由拦截 AudioContext。
  ensurePlaybackContext();
  updateVoiceButton();
  setVoiceHint('正在开启麦克风……');
  try {
    // 录音上下文也必须在手势内同步创建。getUserMedia 是异步的，等它
    // 返回后再 new AudioContext 已经失去用户激活，部分浏览器会让它一直
    // 处于 suspended，ScriptProcessor 收不到任何帧——表现就是“听不到”。
    const context = createCaptureContext();
    voice.context = context;
    context.resume().catch(() => {});
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: true },
        autoGainControl: { ideal: true },
      },
    });
    if (!voice.active || token !== voice.session) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    voice.stream = stream;
    if (context.state === 'suspended') await context.resume();
    if (!voice.active || token !== voice.session) {
      await releaseVoiceCapture();
      return;
    }
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(1024, 1, 1);
    processor.onaudioprocess = (event) => {
      if (token !== voice.session || !voice.active || voice.mode !== 'listening') return;
      handleVoiceFrame(new Float32Array(event.inputBuffer.getChannelData(0)), context.sampleRate);
    };
    const mute = context.createGain();
    mute.gain.value = 0;
    source.connect(processor);
    processor.connect(mute);
    mute.connect(context.destination);
    voice.nodes = [source, processor, mute];
    setVoiceHint(voiceListeningHint(true));
    armNoSpeechReminder();
  } catch (error) {
    if (token !== voice.session) return;
    stopVoiceSession();
    toast(`无法访问麦克风：${error.message || error.name || '未知错误'}`, true);
  }
}

function stopVoiceSession() {
  if (!voice.active) return;
  voice.session += 1;
  voice.active = false;
  voice.mode = 'idle';
  voice.vad = null;
  clearNoSpeechReminder();
  stopVoicePlayback();
  releaseVoiceCapture().catch(() => {});
  updateVoiceButton();
  setVoiceHint();
}

async function toggleVoiceSession() {
  if (voice.active) {
    stopVoiceSession();
    return;
  }
  await startVoiceSession();
}

function scheduleVoicePcm(bytes, sampleRate) {
  const context = ensurePlaybackContext();
  if (!context || !bytes?.length) return;
  let data = bytes;
  if (voice.pcmRemainder.length) {
    const merged = new Uint8Array(voice.pcmRemainder.length + data.length);
    merged.set(voice.pcmRemainder, 0);
    merged.set(data, voice.pcmRemainder.length);
    data = merged;
  }
  // fetch 分块可能落在奇数字节，或复用奇数偏移的底层 buffer。
  const even = data.length - (data.length % 2);
  voice.pcmRemainder = even < data.length ? data.slice(even) : new Uint8Array(0);
  if (!even) return;
  if (data.byteOffset % 2) data = data.slice(0, even);
  const samples = new Int16Array(data.buffer, data.byteOffset, even >> 1);
  const buffer = context.createBuffer(1, samples.length, sampleRate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] / 32768;
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  if (voice.nextStart < context.currentTime + 0.03) {
    voice.nextStart = context.currentTime + 0.03;
  }
  source.start(voice.nextStart);
  voice.nextStart += buffer.duration;
  voice.sources.push(source);
  source.onended = () => {
    const index = voice.sources.indexOf(source);
    if (index >= 0) voice.sources.splice(index, 1);
  };
}

async function speakReply(text) {
  const spoken = String(text || '').slice(0, 3900);
  if (!spoken.trim()) return;
  stopVoicePlayback();
  const controller = new AbortController();
  voice.streamAbort = controller;
  const context = ensurePlaybackContext();
  try {
    if (!context) {
      await playLegacyWav(spoken, controller.signal);
      return;
    }
    if (context.state === 'suspended') {
      try { await context.resume(); } catch { /* 播放时再处理拦截 */ }
    }
    const response = await fetch('/api/voice/speak/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: spoken }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `语音合成失败 (${response.status})`);
    }
    if (!response.body) {
      await playLegacyWav(spoken, controller.signal);
      return;
    }
    const sampleRate = Number(response.headers.get('X-Sample-Rate')) || 24000;
    const reader = response.body.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value?.length) scheduleVoicePcm(value, sampleRate);
      }
    } finally {
      reader.releaseLock?.();
    }
    voice.pcmRemainder = new Uint8Array(0);
  } catch (error) {
    if (error.name === 'AbortError') return;
    stopVoicePlayback();
    throw error;
  } finally {
    if (voice.streamAbort === controller) voice.streamAbort = null;
  }
}

async function playLegacyWav(text, signal) {
  const response = await fetch('/api/voice/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `语音合成失败 (${response.status})`);
  }
  const blob = await response.blob();
  if (!blob.size) return;
  const player = new Audio(URL.createObjectURL(blob));
  voice.player = player;
  player.addEventListener('ended', stopVoicePlayback);
  try {
    await player.play();
  } catch {
    stopVoicePlayback();
    throw new Error('浏览器阻止了语音播放，请点击页面后重试。');
  }
}

function stopVoicePlayback() {
  voice.streamAbort?.abort();
  voice.streamAbort = null;
  for (const source of voice.sources) {
    try { source.stop(); } catch { /* 已结束 */ }
  }
  voice.sources = [];
  voice.nextStart = 0;
  voice.pcmRemainder = new Uint8Array(0);
  if (!voice.player) return;
  const player = voice.player;
  voice.player = null;
  player.pause();
  URL.revokeObjectURL(player.src);
}

async function initVoice() {
  const button = $('voice-btn');
  if (!button) return;
  try {
    const status = await request('/api/voice/status');
    voice.available = Boolean(status.available);
    button.hidden = !voice.available;
  } catch {
    voice.available = false;
    button.hidden = true;
  }
}

const voiceButton = $('voice-btn');
voiceButton.addEventListener('click', (event) => {
  event.preventDefault();
  toggleVoiceSession();
});
voiceButton.addEventListener('contextmenu', (event) => event.preventDefault());

/* ---------------- 事件:模型配置 ---------------- */
function modelFormPayload() {
  const label = $('model-label').value.trim();
  return {
    id: state.editingModelId === NEW_MODEL_ID ? '' : state.editingModelId,
    name: label || $('model-name').value,
    provider: $('model-provider').value, model: $('model-name').value,
    base_url: $('model-base-url').value, api_key: $('model-api-key').value,
    temperature: $('model-temperature').value, max_tokens: $('model-max-tokens').value,
    timeout_seconds: $('model-timeout').value, max_steps: $('model-max-steps').value,
    model_call_timeout_seconds: $('model-call-timeout').value,
    tool_call_timeout_seconds: $('tool-call-timeout').value,
    turn_timeout_seconds: $('turn-timeout').value,
    planning_mode: $('model-planning-mode').value,
    disable_thinking: $('model-disable-thinking').checked,
  };
}

$('model-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.submitter;
  setBusy(button, true, '保存中…');
  showFormError('model-form-error');
  try {
    const saved = await post('/api/config/model/save', modelFormPayload());
    // 新建的 profile 由后端分配 id,拿回来再选中它,否则列表刷新后会跳回当前模型。
    state.editingModelId = saved.model.id;
    $('model-form').dataset.dirty = 'false';
    await loadState();
    $('model-save-msg').textContent = '已保存';
    toast(`模型「${saved.model.name}」已保存。`);
  } catch (error) { showFormError('model-form-error', error.message); toast(error.message, true); }
  finally { setBusy(button, false); }
});

$('add-model-btn').addEventListener('click', () => selectModel(NEW_MODEL_ID));

$('activate-model-btn').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  setBusy(button, true, '切换中…');
  showFormError('model-form-error');
  try {
    const result = await post('/api/config/model/activate', { id: state.editingModelId });
    $('model-form').dataset.dirty = 'false';
    await loadState();
    toast(`已切换到「${result.model.name}」,适配器 ${result.model_adapter}。`);
  } catch (error) { showFormError('model-form-error', error.message); toast(error.message, true); }
  finally { setBusy(button, false); }
});

$('delete-model-btn').addEventListener('click', async (event) => {
  const profile = editingProfile();
  if (!profile) return;
  if (!window.confirm(`删除模型「${profile.name || profile.model}」？该模型的全部参数会一并移除。`)) return;
  const button = event.currentTarget;
  setBusy(button, true, '删除中…');
  showFormError('model-form-error');
  try {
    // 先落后端,成功了再动本地,避免前后端静默分叉。
    await post('/api/config/model/delete', { id: profile.id });
    $('model-form').dataset.dirty = 'false';
    state.editingModelId = '';
    await loadState();
    toast('模型已删除。');
  } catch (error) { showFormError('model-form-error', error.message); toast(error.message, true); }
  finally { setBusy(button, false); }
});

/* ---------------- 事件:MCP / 插件 / 记忆 ---------------- */
$('mcp-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('save-mcp-btn');
  setBusy(button, true, '保存中…');
  showFormError('mcp-form-error');
  try {
    await post('/api/config/mcp', {
      id: $('mcp-id').value, name: $('mcp-name').value, transport: $('mcp-transport').value,
      endpoint: $('mcp-endpoint').value, command: $('mcp-command').value,
      arguments: csv($('mcp-arguments').value), env_keys: csv($('mcp-env-keys').value),
      tool_allowlist: csv($('mcp-tools').value), enabled: $('mcp-enabled').checked,
    });
    $('mcp-dialog').close();
    await loadState();
    toast('MCP Server 声明已保存。');
  } catch (error) { showFormError('mcp-form-error', error.message); toast(error.message, true); }
  finally { setBusy(button, false); }
});

$('plugin-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('save-plugin-btn');
  setBusy(button, true, '保存中…');
  showFormError('plugin-form-error');
  try {
    let config;
    try { config = JSON.parse($('plugin-config').value || '{}'); }
    catch { throw new Error('插件配置必须是有效的 JSON 对象。'); }
    if (!config || Array.isArray(config) || typeof config !== 'object') throw new Error('插件配置必须是 JSON 对象。');
    await post('/api/config/plugin', {
      id: $('plugin-id').value, name: $('plugin-name').value, version: $('plugin-version').value,
      reference: $('plugin-reference').value, config, enabled: $('plugin-enabled').checked,
    });
    $('plugin-dialog').close();
    await loadState();
    toast('插件声明已保存。');
  } catch (error) { showFormError('plugin-form-error', error.message); toast(error.message, true); }
  finally { setBusy(button, false); }
});

/* ---------------- 事件:导航 ---------------- */
function confirmUnsavedModelChanges() {
  return $('model-form').dataset.dirty !== 'true'
    || window.confirm('模型配置尚未保存。放弃更改并离开吗？');
}

function activateView(viewName, { updateHistory = true } = {}) {
  const item = document.querySelector(`.nav-item[data-view="${viewName}"]`);
  if (!item) return;
  document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach((n) => {
    n.removeAttribute('aria-current');
  });
  document.querySelectorAll('.view').forEach((v) => { v.classList.remove('active'); v.hidden = true; });
  // “工作台”代表未接入设备的通用助手；绑定设备时由左侧设备项高亮。
  if (!(viewName === 'console' && MACHINE_ID)) {
    item.classList.add('active');
    item.setAttribute('aria-current', 'page');
  }
  const view = $(`view-${viewName}`);
  view.classList.add('active');
  view.hidden = false;
  state.activeView = viewName;
  if (updateHistory) updateUrl();
}

document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => {
  if (state.activeView === 'settings' && state.activeSettings === 'model' && item.dataset.view !== 'settings' && !confirmUnsavedModelChanges()) return;
  if (item.dataset.view === 'console') {
    selectGeneralAgent();
    return;
  }
  activateView(item.dataset.view);
}));

function activateSettings(settingsName, { updateHistory = true } = {}) {
  const tab = document.querySelector(`.settings-tab[data-settings="${settingsName}"]`);
  if (!tab) return;
  document.querySelectorAll('.settings-tab').forEach((t) => {
    const active = t === tab;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
    t.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.settings-pane').forEach((p) => {
    const active = p.id === `settings-${settingsName}`;
    p.classList.toggle('active', active);
    p.hidden = !active;
  });
  state.activeSettings = settingsName;
  if (updateHistory) updateUrl();
}

document.querySelectorAll('.settings-tab').forEach((tab) => tab.addEventListener('click', () => {
  if (state.activeSettings === 'model' && tab.dataset.settings !== 'model' && !confirmUnsavedModelChanges()) return;
  activateSettings(tab.dataset.settings);
}));
document.querySelectorAll('.settings-tab').forEach((tab) => tab.addEventListener('keydown', (event) => {
  // 标签栏在宽屏是纵向、窄屏转横向,所以上下左右都要认。
  const forward = ['ArrowRight', 'ArrowDown'];
  const backward = ['ArrowLeft', 'ArrowUp'];
  if (![...forward, ...backward, 'Home', 'End'].includes(event.key)) return;
  const tabs = [...document.querySelectorAll('.settings-tab')];
  const current = tabs.indexOf(tab);
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
    : (current + (forward.includes(event.key) ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  if (state.activeSettings === 'model' && tabs[next].dataset.settings !== 'model'
      && !confirmUnsavedModelChanges()) return;
  tabs[next].focus();
  activateSettings(tabs[next].dataset.settings);
}));

/* ---------------- 事件:委托(配置行 / 关闭对话框) ---------------- */
document.addEventListener('click', async (event) => {
  const target = event.target.closest('[data-action], [data-close]');
  if (!target) return;
  if (target.dataset.close) { $(target.dataset.close).close(); return; }
  const id = target.dataset.id;
  const action = target.dataset.action;
  if (action === 'approve-turn' || action === 'reject-turn') {
    const input = $('turn-input');
    input.value = action === 'approve-turn' ? '确认执行' : '取消执行';
    $('turn-form').requestSubmit();
    return;
  }
  if (action === 'edit-model') { selectModel(id); return; }
  if (action === 'edit-mcp') openMcp(state.configuration.mcp_servers.find((s) => s.id === id));
  if (action === 'edit-plugin') openPlugin(state.configuration.plugins.find((p) => p.id === id));
  if (action === 'delete-mcp' && window.confirm('移除此 MCP Server 配置?')) {
    try { await post('/api/config/mcp/delete', { id }); await loadState(); toast('MCP 配置已移除。'); }
    catch (error) { toast(error.message, true); }
  }
  if (action === 'delete-plugin' && window.confirm('移除此插件配置?')) {
    try { await post('/api/config/plugin/delete', { id }); await loadState(); toast('插件配置已移除。'); }
    catch (error) { toast(error.message, true); }
  }
});

/* ---------------- 事件:其它 ---------------- */
$('event-filter').addEventListener('change', (e) => { state.eventFilter = e.target.value; renderEvents(state.events); updateUrl(); });
$('model-form').addEventListener('input', () => { $('model-form').dataset.dirty = 'true'; });
window.addEventListener('beforeunload', (event) => {
  if ($('model-form').dataset.dirty !== 'true') return;
  event.preventDefault();
  event.returnValue = '';
});
$('add-mcp-btn').addEventListener('click', () => openMcp());
$('add-plugin-btn').addEventListener('click', () => openPlugin());
$('refresh-btn').addEventListener('click', () => loadMachines().then(loadState).then(() => toast('已刷新。')).catch((e) => toast(e.message, true)));
$('audit-refresh').addEventListener('click', () => loadState().catch((e) => toast(e.message, true)));

/* ---------------- 初始路由 + 首次加载 ---------------- */
const route = new URLSearchParams(window.location.search);
const initView = ['console', 'machine', 'capabilities', 'settings', 'audit'].includes(route.get('view')) ? route.get('view') : 'console';
const initSettings = ['model', 'mcp', 'plugins'].includes(route.get('settings')) ? route.get('settings') : 'model';
state.eventFilter = route.get('event') || '';
activateView(initView, { updateHistory: false });
activateSettings(initSettings, { updateHistory: false });
window.addEventListener('popstate', () => {
  const current = new URLSearchParams(window.location.search);
  state.eventFilter = current.get('event') || '';
  activateView(['console', 'machine', 'capabilities', 'settings', 'audit'].includes(current.get('view')) ? current.get('view') : 'console', { updateHistory: false });
  activateSettings(['model', 'mcp', 'plugins'].includes(current.get('settings')) ? current.get('settings') : 'model', { updateHistory: false });
  renderEvents(state.events);
});

initVoice();

bootstrapConsole().catch((error) => {
  $('conversation').innerHTML = `<div class="empty-state"><b>无法连接调试服务</b><span>${esc(error.message)}</span></div>`;
});
