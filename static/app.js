const $ = (selector) => document.querySelector(selector);

const ui = {
  newChat: $("#new-chat"), conversationList: $("#conversation-list"), importButton: $("#import-button"), importFile: $("#import-file"),
  chatTitle: $("#chat-title"), subtitle: $("#chat-subtitle"), exportButton: $("#export-button"), deleteChat: $("#delete-chat"),
  messages: $("#messages"), form: $("#message-form"), input: $("#message-input"), send: $("#send-button"), error: $("#form-error"),
  voiceButton: $("#voice-button"), voiceStatus: $("#voice-status"),
  presetSelect: $("#preset-select"), settingsToggle: $("#settings-toggle"), settingsPanel: $("#settings-panel"), settingsClose: $("#settings-close"),
  settingsSummary: $("#settings-summary"), resetCustom: $("#reset-custom"), saveAsPreset: $("#save-as-preset"), presetsButton: $("#presets-button"),
  requestStatus: $("#request-status"), tokenOverview: $("#token-overview"),
  contextMode: $("#context-mode"), contextStatus: $("#context-status"), contextMetrics: $("#context-metrics"),
  summaryCurrent: $("#summary-current"), summaryCurrentMeta: $("#summary-current-meta"), summaryHistory: $("#summary-history"), summaryCount: $("#summary-count"),
  summarySection: $("#summary-section"), summaryHistorySection: $("#summary-history-section"),
  windowSetting: $("#window-setting"), windowExchanges: $("#window-exchanges"), contextRuleText: $("#context-rule-text"),
  factsSection: $("#facts-section"), factsList: $("#facts-list"), factsCount: $("#facts-count"), factForm: $("#fact-form"),
  factKey: $("#fact-key"), factValue: $("#fact-value"), factsError: $("#facts-error"),
  branchingSection: $("#branching-section"), branchCount: $("#branch-count"),
  presetDialog: $("#preset-dialog"), presetList: $("#preset-list"), presetName: $("#preset-name"), presetDescription: $("#preset-description"),
  presetReasoning: $("#preset-reasoning"), presetEditId: $("#preset-edit-id"), presetSave: $("#preset-save"), presetCancelEdit: $("#preset-cancel-edit"), presetError: $("#preset-error"),
  model: $("#setting-model"), system: $("#setting-system"), temperature: $("#setting-temperature"), topP: $("#setting-top-p"),
  reasoning: $("#setting-reasoning"), effort: $("#setting-effort"), maxTokens: $("#setting-max-tokens"), stop: $("#setting-stop"),
  format: $("#setting-format"), logprobs: $("#setting-logprobs"), topLogprobs: $("#setting-top-logprobs"), template: $("#message-template"),
};

let appState = { conversations: [], presets: [], defaults: null, provider: null };
let activeConversation = null;
let selectedPresetId = "";
let sending = false;
let voiceState = { phase: "starting", ready: false, message: "Whisper запускается…" };
let mediaRecorder = null;
let microphoneStream = null;
let audioChunks = [];
let recordingTimer = null;
let voiceBusy = false;

start().catch(showFatal);

async function start() {
  const data = await api("/api/state");
  appState = { conversations: data.conversations, presets: data.presets, defaults: data.default_settings, provider: data.provider };
  populateModels();
  renderPresetSelect();
  applySettings(appState.defaults);
  renderConversationList();
  if (appState.conversations.length) await openConversation(appState.conversations[0].id);
  else await createConversation();
  startVoiceStatusPolling();
}

function startVoiceStatusPolling() {
  refreshVoiceStatus().catch(() => {});
  window.setInterval(() => refreshVoiceStatus().catch(() => {}), 3000);
}

async function refreshVoiceStatus() {
  const data = await api("/api/voice/status");
  voiceState = data.voice;
  renderVoiceStatus();
}

function renderVoiceStatus() {
  if (mediaRecorder?.state === "recording" || voiceBusy) return;
  ui.voiceStatus.textContent = voiceState.message || "Whisper недоступен.";
  ui.voiceStatus.className = `voice-status ${voiceState.phase || "error"}`;
  ui.voiceButton.disabled = sending || !voiceState.ready;
  ui.voiceButton.classList.remove("recording", "processing");
  ui.voiceButton.title = voiceState.ready ? "Надиктовать сообщение" : ui.voiceStatus.textContent;
}

function populateModels() {
  ui.model.replaceChildren();
  appState.provider.models.forEach((model) => ui.model.add(new Option(model, model)));
}

async function createConversation() {
  const data = await api("/api/conversations", { method: "POST", body: { title: "Новый диалог" } });
  await refreshState();
  await openConversation(data.conversation.id);
  ui.input.focus();
}

async function refreshState() {
  const data = await api("/api/state");
  appState.conversations = data.conversations;
  appState.presets = data.presets;
  renderConversationList();
  renderPresetSelect();
}

async function openConversation(id) {
  const data = await api(`/api/conversations/${id}`);
  activeConversation = data.conversation;
  ui.chatTitle.value = activeConversation.title;
  renderMessages();
  renderConversationList();
  const lastUser = [...conversationMessages()].reverse().find((item) => item.role === "user");
  if (lastUser?.technical?.settings) {
    applySettings(lastUser.technical.settings);
    selectedPresetId = lastUser.technical.configuration_source?.preset_id || "";
    if (!appState.presets.some((item) => item.id === selectedPresetId)) selectedPresetId = "";
    ui.presetSelect.value = selectedPresetId;
  }
  updateSettingsSummary();
}

function renderConversationList() {
  ui.conversationList.replaceChildren();
  appState.conversations.forEach((conversation) => {
    const button = document.createElement("button");
    button.className = `conversation-item${activeConversation?.id === conversation.id ? " active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = conversation.title;
    const meta = document.createElement("small");
    meta.textContent = `${conversation.message_count} сообщ. · ${formatDate(conversation.updated_at)}`;
    button.append(title, meta);
    button.addEventListener("click", () => openConversation(conversation.id).catch(showError));
    ui.conversationList.append(button);
  });
}

function renderMessages() {
  ui.messages.replaceChildren();
  renderTokenOverview();
  renderContextPanel();
  const visibleMessages = conversationMessages();
  if (!visibleMessages.length) {
    const welcome = document.createElement("div"); welcome.className = "welcome";
    welcome.innerHTML = '<span class="welcome-mark">D</span><h1>Первый агент готов к диалогу</h1><p>Выберите пресет или настройте один запрос вручную. Контекст и технические данные сохраняются после каждого сообщения.</p>';
    ui.messages.append(welcome); return;
  }
  const tokenIndex = buildTokenIndex(visibleMessages);
  const points = new Map((activeConversation.branch_points || []).map((point) => [point.checkpoint_id, point]));
  visibleMessages.forEach((message) => {
    if (message.role === "event") {
      const event = document.createElement("div"); event.className = "event";
      const span = document.createElement("span"); span.textContent = message.content; event.append(span); ui.messages.append(event); return;
    }
    const element = ui.template.content.firstElementChild.cloneNode(true);
    element.classList.add(message.role);
    if (message.technical?.request_status === "failed") element.classList.add("failed");
    element.querySelector(".message-avatar").textContent = message.role === "user" ? "Вы" : "D";
    element.querySelector("header strong").textContent = message.role === "user" ? "Вы" : "DeepSeek";
    element.querySelector("time").textContent = formatDate(message.created_at, true);
    element.querySelector(".message-text").textContent = message.content;
    const actions = element.querySelector(".message-actions");
    const fork = element.querySelector(".fork-button");
    actions.hidden = activeConversation.context_management?.mode !== "branching";
    fork.addEventListener("click", () => createBranch(message.id));
    const point = points.get(message.id);
    if (point) renderBranchSwitch(element.querySelector(".branch-switch"), point);
    renderMessageTokens(element.querySelector(".token-strip"), message, tokenIndex);
    const reasoning = element.querySelector(".reasoning");
    if (message.reasoning_content) { reasoning.hidden = false; reasoning.querySelector("div").textContent = message.reasoning_content; }
    const technical = element.querySelector(".technical");
    technical.querySelector("pre").textContent = JSON.stringify(message.technical || {}, null, 2);
    ui.messages.append(element);
  });
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function conversationMessages() {
  if (!activeConversation) return [];
  return Array.isArray(activeConversation.visible_messages) ? activeConversation.visible_messages : (activeConversation.messages || []);
}

function renderBranchSwitch(container, point) {
  container.hidden = false;
  const previous = container.querySelector(".branch-prev");
  const next = container.querySelector(".branch-next");
  container.querySelector("span").textContent = `${point.active_index + 1} / ${point.options.length}`;
  previous.disabled = point.active_index <= 0;
  next.disabled = point.active_index >= point.options.length - 1;
  previous.addEventListener("click", () => switchBranch(point, point.active_index - 1));
  next.addEventListener("click", () => switchBranch(point, point.active_index + 1));
}

function renderContextPanel() {
  if (!activeConversation) return;
  const context = activeConversation.context_management || { mode: "full", keep_recent_exchanges: 5, summary_batch_exchanges: 5 };
  const summaries = Array.isArray(activeConversation.summaries) ? activeConversation.summaries : [];
  const current = [...summaries].reverse().find((item) => item.id === context.active_summary_id) || null;
  const summaryTotals = activeConversation.summary_token_totals || {};
  const factsTotals = activeConversation.facts_token_totals || {};
  const completed = (activeConversation.active_path_token_totals?.request_count || 0);
  const covered = Number(current?.covered_exchange_count) || 0;
  const labels = { full: "Полная история", summary: "Summary", sliding: "Sliding Window", facts: "Sticky Facts", branching: "Branching" };
  const rules = {
    full: "В запрос отправляется вся успешная история активного пути.",
    summary: "Последние 5 обменов остаются дословно; каждые 5 старых обменов сворачиваются моделью deepseek-v4-flash.",
    sliding: `В запрос отправляются только последние ${context.sliding_window_exchanges || 5} полных обменов.`,
    facts: `Facts обновляются после каждого сообщения пользователя и отправляются вместе с последними ${context.facts_window_exchanges || 5} обменами.`,
    branching: "Каждая ветка — независимый полный путь. Разветвление возможно после любого сообщения.",
  };

  ui.contextMode.value = context.mode;
  ui.contextStatus.textContent = labels[context.mode] || labels.full;
  ui.contextStatus.classList.toggle("enabled", context.mode !== "full");
  ui.contextRuleText.textContent = rules[context.mode] || rules.full;
  ui.windowSetting.hidden = !["sliding", "facts"].includes(context.mode);
  ui.windowExchanges.value = context.mode === "facts" ? (context.facts_window_exchanges || 5) : (context.sliding_window_exchanges || 5);
  ui.summarySection.hidden = context.mode !== "summary";
  ui.summaryHistorySection.hidden = context.mode !== "summary";
  ui.factsSection.hidden = context.mode !== "facts";
  ui.branchingSection.hidden = context.mode !== "branching";
  ui.contextMetrics.replaceChildren();
  addContextMetric("Обменов в пути", completed);
  if (context.mode === "summary") {
    addContextMetric("Свёрнуто", covered);
    addContextMetric("Версий", summaries.length);
    addContextMetric("Токены summary", summaryTotals.total_tokens || 0);
  } else if (context.mode === "facts") {
    addContextMetric("Фактов", (activeConversation.facts || []).length);
    addContextMetric("Обновлений", factsTotals.request_count || 0);
    addContextMetric("Токены facts", factsTotals.total_tokens || 0);
  } else if (context.mode === "branching") {
    addContextMetric("Развилок в пути", (activeConversation.branch_points || []).length);
    addContextMetric("Всего сообщений", (activeConversation.messages || []).filter((item) => ["user", "assistant"].includes(item.role)).length);
  } else if (context.mode === "sliding") {
    addContextMetric("Размер окна", context.sliding_window_exchanges || 5);
  }

  ui.summaryCurrent.textContent = current?.content || (context.mode === "summary"
    ? "Summary появится, когда за последними 5 обменами накопятся ещё 5 старых."
    : "Сжатие выключено. Ранее созданные версии остаются в JSON.");
  ui.summaryCurrentMeta.textContent = current ? `${current.covered_exchange_count} обменов · ${formatDate(current.created_at, true)}` : "";
  ui.summaryCount.textContent = `${summaries.length} ${plural(summaries.length, "версия", "версии", "версий")}`;
  ui.summaryHistory.replaceChildren();
  if (!summaries.length) {
    const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Сохранённых версий пока нет."; ui.summaryHistory.append(empty);
  } else {
    [...summaries].reverse().forEach((summary, index) => {
      const details = document.createElement("details"); details.className = "summary-version"; details.open = index === 0;
      const heading = document.createElement("summary");
      const label = document.createElement("strong"); label.textContent = `Версия ${summaries.length - index}`;
      const meta = document.createElement("small"); meta.textContent = `${summary.covered_exchange_count || 0} обменов · ${formatDate(summary.created_at, true)}`;
      heading.append(label, meta);
      const content = document.createElement("div"); content.className = "summary-version-text"; content.textContent = summary.content || "";
      const usage = document.createElement("div"); usage.className = "summary-usage";
      const tokens = summary.technical?.usage || {};
      usage.textContent = `Вход ${number(tokens.input_tokens)} · выход ${number(tokens.output_tokens)} · всего ${number(tokens.total_tokens)}`;
      details.append(heading, content, usage); ui.summaryHistory.append(details);
    });
  }
  renderFacts();
  ui.branchCount.textContent = `${(activeConversation.branch_points || []).length} развилок`;
}

function renderFacts() {
  const facts = Array.isArray(activeConversation?.facts) ? activeConversation.facts : [];
  ui.factsCount.textContent = `${facts.length} ${plural(facts.length, "факт", "факта", "фактов")}`;
  ui.factsList.replaceChildren();
  if (!facts.length) {
    const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Facts пока не сохранены.";
    ui.factsList.append(empty);
    return;
  }
  facts.forEach((fact) => {
    const row = document.createElement("div"); row.className = "fact-row";
    const key = document.createElement("input"); key.value = fact.key; key.maxLength = 120;
    const value = document.createElement("textarea"); value.value = fact.value; value.maxLength = 2000; value.rows = 2;
    const state = document.createElement("small"); state.textContent = fact.locked ? "🔒 закреплён" : "автоматический";
    const actions = document.createElement("div"); actions.className = "fact-actions";
    actions.append(
      button("Сохранить", "secondary", () => saveFact(fact.id, key.value, value.value, true)),
      button(fact.locked ? "Открепить" : "Закрепить", "secondary", () => saveFact(fact.id, key.value, value.value, !fact.locked)),
      button("×", "danger", () => removeFact(fact.id)),
    );
    row.append(key, value, state, actions); ui.factsList.append(row);
  });
}

function addContextMetric(label, value) {
  const item = document.createElement("div");
  const caption = document.createElement("small"); caption.textContent = label;
  const numberElement = document.createElement("strong"); numberElement.textContent = number(value);
  item.append(caption, numberElement); ui.contextMetrics.append(item);
}

function plural(value, one, few, many) {
  const numberValue = Math.abs(Number(value)) % 100;
  const last = numberValue % 10;
  if (numberValue > 10 && numberValue < 20) return many;
  if (last === 1) return one;
  if (last > 1 && last < 5) return few;
  return many;
}

function number(value) { return new Intl.NumberFormat("ru-RU").format(Number(value) || 0); }

function addTokenBadge(container, label, value, title = "") {
  const badge = document.createElement("span");
  badge.className = "token-badge"; badge.title = title;
  const caption = document.createElement("small"); caption.textContent = label;
  const count = document.createElement("strong"); count.textContent = number(value);
  badge.append(caption, count); container.append(badge);
}

function normalizedUsage(message) {
  const usage = message?.technical?.usage || {};
  const input = Number(usage.input_tokens) || 0;
  const output = Number(usage.output_tokens) || 0;
  return {
    input_tokens: input, output_tokens: output,
    total_tokens: Number(usage.total_tokens) || input + output,
    cached_input_tokens: Number(usage.cached_input_tokens) || 0,
    uncached_input_tokens: Number(usage.uncached_input_tokens) || 0,
    reasoning_tokens: Number(usage.reasoning_tokens) || 0,
  };
}

function buildTokenIndex(messages) {
  const exchanges = new Map();
  const cumulative = { input_tokens: 0, output_tokens: 0, total_tokens: 0, cached_input_tokens: 0, uncached_input_tokens: 0, reasoning_tokens: 0, request_count: 0 };
  messages.forEach((message) => {
    if (message.role !== "assistant" || !message.technical?.usage || message.technical?.request_status === "failed") return;
    const usage = normalizedUsage(message);
    Object.keys(usage).forEach((key) => { cumulative[key] += usage[key]; });
    cumulative.request_count += 1;
    exchanges.set(message.exchange_id, { usage, cumulative: { ...cumulative } });
  });
  return { exchanges, totals: cumulative };
}

function renderMessageTokens(container, message, tokenIndex) {
  const exchange = tokenIndex.exchanges.get(message.exchange_id);
  if (!exchange) return;
  container.hidden = false;
  if (message.role === "user") {
    addTokenBadge(container, "Контекст вызова", exchange.usage.input_tokens, "Системный промпт, предыдущая история и этот запрос");
    addTokenBadge(container, "Из кэша", exchange.usage.cached_input_tokens, "Входные токены, найденные в кэше DeepSeek");
    addTokenBadge(container, "Без кэша", exchange.usage.uncached_input_tokens, "Входные токены, не найденные в кэше DeepSeek");
  } else if (message.role === "assistant") {
    addTokenBadge(container, "Ответ модели", exchange.usage.output_tokens, "Все выходные токены текущего вызова");
    addTokenBadge(container, "Reasoning", exchange.usage.reasoning_tokens, "Reasoning-токены по детализации API");
    addTokenBadge(container, "Всего за вызов", exchange.usage.total_tokens, "Входные и выходные токены текущего вызова");
    addTokenBadge(container, "Σ диалога", exchange.cumulative.total_tokens, "Сумма токенов всех вызовов к этому месту диалога");
  }
}

function renderTokenOverview() {
  ui.tokenOverview.replaceChildren();
  const totals = buildTokenIndex(conversationMessages()).totals;
  if (!totals.request_count) { ui.tokenOverview.hidden = true; return; }
  ui.tokenOverview.hidden = false;
  addTokenBadge(ui.tokenOverview, "Σ диалога", totals.total_tokens, "Включает повторную отправку истории в каждом вызове");
  addTokenBadge(ui.tokenOverview, "Σ контекст", totals.input_tokens);
  addTokenBadge(ui.tokenOverview, "Σ ответы", totals.output_tokens);
  addTokenBadge(ui.tokenOverview, "Σ кэш", totals.cached_input_tokens);
  addTokenBadge(ui.tokenOverview, "Вызовов", totals.request_count);
}

function renderPresetSelect() {
  const current = selectedPresetId;
  ui.presetSelect.replaceChildren(new Option("Разовые настройки", ""));
  appState.presets.forEach((preset) => ui.presetSelect.add(new Option(preset.name, preset.id)));
  ui.presetSelect.value = appState.presets.some((item) => item.id === current) ? current : "";
}

function renderPresetList() {
  ui.presetList.replaceChildren();
  if (!appState.presets.length) { const empty = document.createElement("div"); empty.className = "empty-note"; empty.textContent = "Сохранённых пресетов пока нет."; ui.presetList.append(empty); return; }
  appState.presets.forEach((preset) => {
    const row = document.createElement("div"); row.className = "preset-row";
    const info = document.createElement("div"); const name = document.createElement("strong"); name.textContent = preset.name;
    const description = document.createElement("p");
    const details = `${preset.settings.model} · t=${preset.settings.temperature} · Reasoning: ${preset.settings.reasoning_enabled ? "Да" : "Нет"}`;
    description.textContent = preset.description ? `${preset.description} · ${details}` : details;
    info.append(name, description);
    const actions = document.createElement("div"); actions.className = "actions";
    const use = button("Выбрать", "secondary", () => {
      selectPreset(preset.id);
      ui.presetDialog.close();
      ui.settingsPanel.classList.remove("open");
    });
    const edit = button("Изменить", "secondary", () => editPreset(preset));
    const remove = button("×", "danger", () => deletePreset(preset.id));
    actions.append(use, edit, remove); row.append(info, actions); ui.presetList.append(row);
  });
}

function button(label, className, handler) { const value = document.createElement("button"); value.type = "button"; value.className = className; value.textContent = label; value.addEventListener("click", handler); return value; }

function selectPreset(id) {
  selectedPresetId = id;
  ui.presetSelect.value = id;
  const preset = appState.presets.find((item) => item.id === id);
  if (preset) applySettings(preset.settings);
  else applySettings(appState.defaults);
  updateSettingsSummary();
}

function applySettings(settings) {
  ui.model.value = settings.model; ui.system.value = settings.system_prompt;
  ui.temperature.value = settings.temperature; ui.topP.value = settings.top_p;
  ui.reasoning.value = String(settings.reasoning_enabled); ui.effort.value = settings.reasoning_effort;
  ui.maxTokens.value = settings.max_tokens ?? ""; ui.stop.value = (settings.stop || []).join("\n");
  ui.format.value = settings.response_format; ui.logprobs.checked = settings.logprobs;
  ui.topLogprobs.value = settings.top_logprobs ?? "";
  updateSettingsSummary();
}

function readSettings() {
  return {
    model: ui.model.value, system_prompt: ui.system.value.trim(), temperature: Number(ui.temperature.value), top_p: Number(ui.topP.value),
    reasoning_enabled: ui.reasoning.value === "true", reasoning_effort: ui.effort.value,
    max_tokens: ui.maxTokens.value === "" ? null : Number(ui.maxTokens.value),
    stop: ui.stop.value.split("\n").map((value) => value.trim()).filter(Boolean), response_format: ui.format.value,
    logprobs: ui.logprobs.checked, top_logprobs: ui.topLogprobs.value === "" ? null : Number(ui.topLogprobs.value),
  };
}

function updateSettingsSummary() {
  if (!appState.defaults) return;
  const value = readSettings();
  const preset = appState.presets.find((item) => item.id === selectedPresetId);
  const changed = preset && JSON.stringify(value) !== JSON.stringify(preset.settings);
  const source = preset ? `${preset.name}${changed ? " · изменён" : ""}` : "Разовые";
  ui.settingsSummary.textContent = `${source} · ${value.model} · t=${value.temperature} · reasoning ${value.reasoning_enabled ? "on" : "off"}`;
}

async function submitMessage(event) {
  event.preventDefault(); if (sending || !activeConversation) return;
  if (mediaRecorder?.state === "recording" || voiceBusy) { ui.error.textContent = "Сначала завершите запись и распознавание."; return; }
  const content = ui.input.value.trim(); if (!content) return;
  sending = true; setSendingState(true); ui.error.textContent = "";
  const pending = renderPendingExchange(content);
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/messages`, { method: "POST", body: { content, preset_id: selectedPresetId || null, settings: readSettings() } });
    activeConversation = data.conversation; ui.input.value = ""; resizeInput(); await refreshState(); renderMessages(); ui.chatTitle.value = activeConversation.title;
  } catch (error) {
    if (error.data?.conversation) { activeConversation = error.data.conversation; renderMessages(); await refreshState(); }
    showError(error);
  } finally { sending = false; setSendingState(false); pending.forEach((element) => element.remove()); ui.input.focus(); }
}

function renderPendingExchange(content) {
  const welcome = ui.messages.querySelector(".welcome"); if (welcome) welcome.remove();
  const user = ui.template.content.firstElementChild.cloneNode(true); user.classList.add("user", "pending-message");
  user.querySelector(".message-avatar").textContent = "Вы"; user.querySelector("header strong").textContent = "Вы";
  user.querySelector("time").textContent = "только что"; ui.messages.append(user);
  user.querySelector(".message-text").textContent = content;
  user.querySelector(".technical").remove(); user.querySelector(".reasoning").remove(); user.querySelector(".message-actions").remove();

  const waiting = document.createElement("article"); waiting.className = "message assistant waiting-message";
  waiting.innerHTML = '<div class="message-avatar">D</div><div class="message-body"><header><strong>DeepSeek</strong><span class="sent-badge">Запрос получен</span></header><div class="thinking-line"><i></i><i></i><i></i><span>Формирует ответ…</span></div></div>';
  ui.messages.append(waiting); ui.messages.scrollTop = ui.messages.scrollHeight;
  return [user, waiting];
}

function setSendingState(active) {
  ui.send.disabled = active; ui.input.disabled = active; ui.form.classList.toggle("sending", active);
  ui.send.textContent = active ? "…" : "↑"; ui.requestStatus.hidden = !active;
  ui.settingsToggle.disabled = active; ui.presetSelect.disabled = active;
  ui.contextMode.disabled = active;
  ui.windowExchanges.disabled = active;
  ui.voiceButton.disabled = active || !voiceState.ready;
}

async function toggleVoiceRecording() {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!voiceState.ready || voiceBusy) return;
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    ui.error.textContent = "Этот браузер не поддерживает запись с микрофона.";
    return;
  }
  ui.error.textContent = "";
  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    const mimeType = candidates.find((item) => MediaRecorder.isTypeSupported(item)) || "";
    mediaRecorder = mimeType ? new MediaRecorder(microphoneStream, { mimeType }) : new MediaRecorder(microphoneStream);
    audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", (event) => { if (event.data.size) audioChunks.push(event.data); });
    mediaRecorder.addEventListener("stop", finishVoiceRecording, { once: true });
    mediaRecorder.start(250);
    ui.voiceButton.classList.add("recording");
    ui.voiceButton.disabled = false;
    ui.voiceButton.title = "Остановить запись";
    ui.voiceStatus.textContent = "Идёт запись… нажмите микрофон для остановки";
    ui.voiceStatus.className = "voice-status recording";
    recordingTimer = window.setTimeout(() => { if (mediaRecorder?.state === "recording") mediaRecorder.stop(); }, 5 * 60 * 1000);
  } catch (error) {
    stopMicrophoneTracks();
    ui.error.textContent = error?.name === "NotAllowedError"
      ? "Доступ к микрофону запрещён. Разрешите его для локальной страницы."
      : `Не удалось включить микрофон: ${error.message || error}`;
    renderVoiceStatus();
  }
}

async function finishVoiceRecording() {
  window.clearTimeout(recordingTimer);
  stopMicrophoneTracks();
  const mimeType = mediaRecorder?.mimeType || "audio/webm";
  const chunks = audioChunks;
  mediaRecorder = null;
  audioChunks = [];
  if (!chunks.length) { ui.error.textContent = "Микрофон не записал звук."; renderVoiceStatus(); return; }

  voiceBusy = true;
  ui.voiceButton.disabled = true;
  ui.voiceButton.classList.remove("recording");
  ui.voiceButton.classList.add("processing");
  ui.voiceStatus.textContent = "Whisper распознаёт запись на RX 6600…";
  ui.voiceStatus.className = "voice-status processing";
  try {
    const extension = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "mp4" : "webm";
    const body = new FormData();
    body.append("audio", new Blob(chunks, { type: mimeType }), `recording.${extension}`);
    const response = await fetch("/api/voice/transcribe", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.error || `Ошибка HTTP ${response.status}`);
    insertRecognizedText(data.text);
    ui.voiceStatus.textContent = "Текст распознан — проверьте и отправьте";
    ui.voiceStatus.className = "voice-status success";
  } catch (error) {
    ui.error.textContent = error.message || String(error);
  } finally {
    voiceBusy = false;
    ui.voiceButton.classList.remove("processing");
    ui.voiceButton.disabled = sending || !voiceState.ready;
    ui.input.focus();
  }
}

function stopMicrophoneTracks() {
  if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
  microphoneStream = null;
}

function insertRecognizedText(text) {
  const start = ui.input.selectionStart ?? ui.input.value.length;
  const end = ui.input.selectionEnd ?? start;
  const before = ui.input.value.slice(0, start);
  const after = ui.input.value.slice(end);
  const prefix = before && !/\s$/.test(before) ? " " : "";
  const suffix = after && !/^\s/.test(after) ? " " : "";
  ui.input.value = `${before}${prefix}${String(text).trim()}${suffix}${after}`;
  const cursor = before.length + prefix.length + String(text).trim().length;
  ui.input.setSelectionRange(cursor, cursor);
  resizeInput();
}

async function changeContextMode() {
  if (!activeConversation || sending) return;
  const requestedMode = ui.contextMode.value;
  ui.contextMode.disabled = true; ui.error.textContent = "";
  try {
    const size = Math.max(1, Math.min(50, Number(ui.windowExchanges.value) || 5));
    const body = { mode: requestedMode };
    if (requestedMode === "sliding") body.sliding_window_exchanges = size;
    if (requestedMode === "facts") body.facts_window_exchanges = size;
    const data = await api(`/api/conversations/${activeConversation.id}/context`, { method: "PATCH", body });
    activeConversation = data.conversation;
    renderMessages();
    await refreshState();
  } catch (error) {
    ui.contextMode.value = activeConversation.context_management?.mode || "full";
    showError(error);
  } finally { ui.contextMode.disabled = false; }
}

async function createBranch(messageId) {
  if (!activeConversation || sending) return;
  sending = true; setSendingState(true); ui.error.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/branches`, {
      method: "POST", body: { checkpoint_id: messageId },
    });
    activeConversation = data.conversation;
    renderMessages();
    ui.input.focus();
  } catch (error) {
    showError(error);
  } finally {
    sending = false; setSendingState(false);
  }
}

async function switchBranch(point, index) {
  if (!activeConversation || sending || index < 0 || index >= point.options.length) return;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/branches/active`, {
      method: "PATCH",
      body: { checkpoint_id: point.checkpoint_id, child_id: point.options[index].child_id },
    });
    activeConversation = data.conversation;
    renderMessages();
  } catch (error) { showError(error); }
}

async function addManualFact(event) {
  event.preventDefault();
  if (!activeConversation) return;
  ui.factsError.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts`, {
      method: "POST", body: { key: ui.factKey.value, value: ui.factValue.value },
    });
    activeConversation = data.conversation;
    ui.factKey.value = ""; ui.factValue.value = "";
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function saveFact(id, key, value, locked) {
  ui.factsError.textContent = "";
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts/${id}`, {
      method: "PUT", body: { key, value, locked },
    });
    activeConversation = data.conversation;
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function removeFact(id) {
  if (!confirm("Удалить этот факт? Он сможет появиться снова при следующем автоматическом обновлении.")) return;
  try {
    const data = await api(`/api/conversations/${activeConversation.id}/facts/${id}`, { method: "DELETE" });
    activeConversation = data.conversation;
    renderContextPanel();
  } catch (error) { ui.factsError.textContent = error.message; }
}

async function renameChat() {
  if (!activeConversation) return;
  try { const data = await api(`/api/conversations/${activeConversation.id}`, { method: "PATCH", body: { title: ui.chatTitle.value } }); activeConversation = data.conversation; await refreshState(); }
  catch (error) { showError(error); }
}

async function deleteChat() {
  if (!activeConversation || !confirm(`Удалить диалог «${activeConversation.title}»?`)) return;
  await api(`/api/conversations/${activeConversation.id}`, { method: "DELETE" }); activeConversation = null; await refreshState();
  if (appState.conversations.length) await openConversation(appState.conversations[0].id); else await createConversation();
}

function openPresetDialog(forSave = false) { renderPresetList(); clearPresetEditor(); if (forSave) ui.presetName.focus(); ui.presetDialog.showModal(); }
function clearPresetEditor() { ui.presetEditId.value = ""; ui.presetName.value = ""; ui.presetDescription.value = ""; ui.presetReasoning.value = ui.reasoning.value; ui.presetError.textContent = ""; $("#preset-editor-title").textContent = "Новый пресет из текущих настроек"; }
function editPreset(preset) { ui.presetEditId.value = preset.id; ui.presetName.value = preset.name; ui.presetDescription.value = preset.description; ui.presetReasoning.value = String(preset.settings.reasoning_enabled); applySettings(preset.settings); $("#preset-editor-title").textContent = "Изменить пресет"; }

async function savePreset() {
  ui.presetError.textContent = "";
  const presetSettings = readSettings(); presetSettings.reasoning_enabled = ui.presetReasoning.value === "true";
  const body = { name: ui.presetName.value, description: ui.presetDescription.value, settings: presetSettings };
  const id = ui.presetEditId.value;
  try {
    const data = await api(id ? `/api/presets/${id}` : "/api/presets", { method: id ? "PUT" : "POST", body });
    selectedPresetId = data.preset.id;
    applySettings(data.preset.settings);
    await refreshState();
    clearPresetEditor();
    ui.presetSelect.value = selectedPresetId;
    updateSettingsSummary();
    ui.presetDialog.close();
    ui.settingsPanel.classList.remove("open");
  } catch (error) { ui.presetError.textContent = error.message; }
}

async function deletePreset(id) {
  const preset = appState.presets.find((item) => item.id === id); if (!confirm(`Удалить пресет «${preset?.name || ""}»?`)) return;
  await api(`/api/presets/${id}`, { method: "DELETE" }); if (selectedPresetId === id) selectedPresetId = ""; await refreshState(); renderPresetList(); updateSettingsSummary();
}

async function importFile(file) {
  try { const bundle = JSON.parse(await file.text()); const data = await api("/api/import", { method: "POST", body: bundle }); await refreshState(); await openConversation(data.conversation.id); }
  catch (error) { showError(error); } finally { ui.importFile.value = ""; }
}

async function api(url, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(options.body); }
  const response = await fetch(url, init); const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) { const error = new Error(data.error || `Ошибка HTTP ${response.status}`); error.data = data; throw error; }
  return data;
}

function formatDate(value, withTime = false) { if (!value) return ""; const date = new Date(value); return date.toLocaleString("ru-RU", withTime ? { dateStyle: "short", timeStyle: "short" } : { day: "2-digit", month: "2-digit" }); }
function showError(error) { ui.error.textContent = error.message || String(error); }
function showFatal(error) { ui.messages.textContent = `Не удалось запустить интерфейс: ${error.message || error}`; }
function resizeInput() { ui.input.style.height = "auto"; ui.input.style.height = `${Math.min(ui.input.scrollHeight, 180)}px`; }

ui.newChat.addEventListener("click", () => createConversation().catch(showError));
ui.form.addEventListener("submit", submitMessage); ui.input.addEventListener("input", resizeInput);
ui.voiceButton.addEventListener("click", toggleVoiceRecording);
ui.input.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ui.form.requestSubmit(); } });
ui.chatTitle.addEventListener("change", renameChat); ui.chatTitle.addEventListener("blur", renameChat);
ui.deleteChat.addEventListener("click", () => deleteChat().catch(showError));
ui.exportButton.addEventListener("click", () => { if (activeConversation) window.location.href = `/api/conversations/${activeConversation.id}/export`; });
ui.importButton.addEventListener("click", () => ui.importFile.click()); ui.importFile.addEventListener("change", () => { if (ui.importFile.files[0]) importFile(ui.importFile.files[0]); });
ui.settingsToggle.addEventListener("click", () => ui.settingsPanel.classList.add("open")); ui.settingsClose.addEventListener("click", () => ui.settingsPanel.classList.remove("open"));
ui.contextMode.addEventListener("change", changeContextMode);
ui.windowExchanges.addEventListener("change", changeContextMode);
ui.factForm.addEventListener("submit", addManualFact);
ui.presetSelect.addEventListener("change", () => {
  selectPreset(ui.presetSelect.value);
  if (!ui.presetSelect.value) ui.settingsPanel.classList.add("open");
  else ui.settingsPanel.classList.remove("open");
});
ui.resetCustom.addEventListener("click", () => { selectedPresetId = ""; ui.presetSelect.value = ""; applySettings(appState.defaults); });
ui.saveAsPreset.addEventListener("click", () => openPresetDialog(true)); ui.presetsButton.addEventListener("click", () => openPresetDialog());
ui.presetSave.addEventListener("click", savePreset); ui.presetCancelEdit.addEventListener("click", clearPresetEditor);
document.querySelectorAll("#settings-panel input, #settings-panel textarea, #settings-panel select").forEach((control) => control.addEventListener("input", updateSettingsSummary));
