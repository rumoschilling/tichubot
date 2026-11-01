"use strict";

// The critic is trained on team score difference / 400.  Keep that stable
// training representation on the wire, but present the useful point estimate.
const VALUE_POINT_SCALE = 400;

function loadSettings() {
  const revealDelays = [0, 250, 500, 1000, 1500];
  try {
    const stored = JSON.parse(window.localStorage.getItem("tichubot-settings") || "{}");
    return {
      passingStyle: stored.passingStyle === "slots" ? "slots" : "ordered",
      showAiSuggestions: stored.showAiSuggestions !== false,
      aiRevealDelayMs: revealDelays.includes(stored.aiRevealDelayMs)
        ? stored.aiRevealDelayMs
        : 0,
      replayRevealDelayMs: revealDelays.includes(stored.replayRevealDelayMs)
        ? stored.replayRevealDelayMs
        : 500,
    };
  } catch (_) {
    return {
      passingStyle: "ordered",
      showAiSuggestions: true,
      aiRevealDelayMs: 0,
      replayRevealDelayMs: 500,
    };
  }
}

const state = {
  game: null,
  selected: [],
  passCards: [null, null, null],
  passSlot: 0,
  gtCallers: [],
  settings: loadSettings(),
  showingSettings: false,
  replayPlaying: true,
  error: "",
};
const cardTemplate = document.querySelector("#card-template");
let aiTimer = null;
let aiRequestInFlight = false;

function stopAiSchedule() {
  if (aiTimer !== null) window.clearTimeout(aiTimer);
  aiTimer = null;
}

async function advanceAiStep(game) {
  const replay = Boolean(game?.spectating);
  if (aiRequestInFlight) return;
  if (replay && !game.replay_can_forward) return;
  if (!replay && !Number.isInteger(game?.ai_action_id)) return;
  aiRequestInFlight = true;
  try {
    // The server has already run inference and bound this id to one exact
    // action.  Sending it back prevents an old timer from revealing a newer
    // opponent move after the table has advanced.
    await request(
      replay ? "/api/replay/forward" : "/api/advance-ai",
      replay ? {} : { action_id: game.ai_action_id },
    );
  } finally {
    aiRequestInFlight = false;
    scheduleAiStep(state.game);
  }
}

function scheduleAiStep(game) {
  stopAiSchedule();
  const canAdvance = game?.spectating ? game.replay_can_forward : game?.ai_pending;
  if (!canAdvance || state.showingSettings || aiRequestInFlight) return;
  if (game.spectating && !state.replayPlaying) return;
  const delay = aiRevealDelay(game);
  aiTimer = window.setTimeout(() => {
    aiTimer = null;
    advanceAiStep(game);
  }, Math.max(0, delay));
}

function aiPlayer(game) {
  // Fallback makes a newly refreshed UI compatible with an older local server
  // process until the user restarts it.
  return Number.isInteger(game?.ai_player) ? game.ai_player : game?.current_player;
}

function aiRevealDelay(game) {
  const serverDelay = Number.isFinite(game?.ai_delay_ms) ? game.ai_delay_ms : 500;
  if (serverDelay <= 0) return 0;
  if (game?.spectating) return state.settings.replayRevealDelayMs;
  // Your partner is playing for your team, so reveal their already-selected
  // move immediately. The setting deliberately paces opponents only.
  const player = aiPlayer(game);
  const isPartner = Number.isInteger(player)
    && Number.isInteger(game?.seat)
    && player !== game.seat
    && player % 2 === game.seat % 2;
  return !isPartner ? state.settings.aiRevealDelayMs : 0;
}

async function request(path, body = undefined) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  try {
    const response = await fetch(path, options);
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "The action was not accepted.");
    state.game = payload.state;
    state.error = "";
    state.selected = [];
    state.gtCallers = [];
    resetPassSelection();
  } catch (error) {
    state.error = error.message || "The local server could not be reached.";
  }
  render();
}

function spriteStyle(card) {
  // Source sheet preview geometry: a 1277×721 render has 90×137 faces at
  // x=10+98*column and y=7+143*row.  Scale it to this card's CSS width.
  const width = card.getBoundingClientRect().width;
  const scale = width / 90;
  const col = Number(card.dataset.spriteCol);
  const row = Number(card.dataset.spriteRow);
  card.style.backgroundSize = `${1277 * scale}px ${721 * scale}px`;
  card.style.backgroundPosition = `${-(10 + 98 * col) * scale}px ${-(7 + 143 * row) * scale}px`;
}

function refreshSprites(root = document) {
  root.querySelectorAll(".card[data-sprite-row]").forEach(spriteStyle);
}

function cardElement(card, { selectable = false, selected = false, onClick = null } = {}) {
  const node = cardTemplate.content.firstElementChild.cloneNode(true);
  const receivedFrom = card.received_from;
  node.dataset.index = card.index;
  node.dataset.spriteRow = card.sprite_row;
  node.dataset.spriteCol = card.sprite_col;
  node.title = receivedFrom ? `${card.name} — ${receivedFrom.description}` : card.name;
  node.setAttribute(
    "aria-label",
    receivedFrom ? `${card.name}, ${receivedFrom.description}` : card.name,
  );
  node.classList.toggle("special", Boolean(card.special));
  node.classList.toggle("selected", selected);
  node.disabled = !selectable;
  node.querySelector(".card-label").textContent = card.special ? card.name : "";
  if (receivedFrom) {
    const marker = document.createElement("span");
    marker.className = `pass-source-marker pass-source-${receivedFrom.relation}`;
    marker.textContent = receivedFrom.label;
    marker.title = receivedFrom.description;
    marker.setAttribute("aria-hidden", "true");
    node.append(marker);
  }
  if (onClick) node.addEventListener("click", onClick);
  requestAnimationFrame(() => spriteStyle(node));
  return node;
}

function cardsElement(cards, options = {}) {
  const out = document.createElement("div");
  out.className = `cards ${options.className || ""}`;
  const selectedIndices = options.selectedIndices ?? state.selected;
  cards.forEach((card) => out.append(cardElement(card, {
    selectable: options.selectable,
    selected: selectedIndices.includes(card.index),
    onClick: options.onCard ? () => options.onCard(card.index) : null,
  })));
  return out;
}

function makeButton(label, handler, className = "button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

async function exitApp() {
  stopAiSchedule();
  try {
    await fetch("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  } finally {
    document.body.innerHTML = "<main class=\"app-shell\"><section class=\"setup\"><h1>TichuBot closed</h1><p>You can close this browser tab.</p></section></main>";
  }
}

function cardNames(cards) {
  return cards.map((card) => card.name).join(" ");
}

function predictedValueText(game) {
  if (game.spectating) return game.round_over ? scoreText(game) : "AI replay in progress";
  if (!game.has_model_value) return "Model estimate: unavailable (random AI)";
  if (!game.value_available) return "Model estimate: recalculates on your turn";
  const value = game.predicted_value;
  if (typeof value !== "number" || !Number.isFinite(value)) return "Model estimate (your team): unavailable";
  const points = value * VALUE_POINT_SCALE;
  return `Expected team score difference: ${points >= 0 ? "+" : ""}${points.toFixed(0)} pts`;
}

function scoreText(game) {
  const signed = (value) => `${value >= 0 ? "+" : ""}${value}`;
  return `Final score: Team A ${signed(game.scores[0])} · Team B ${signed(game.scores[1])}`;
}

function renderHeader(game) {
  document.querySelector(".topbar h1").textContent = game.show_mode_menu
    ? "TichuBot"
    : game.mode === "advisor" ? "Advisor" : "Visual table";
  const controls = document.querySelector("#round-controls");
  controls.replaceChildren();
  if (game.show_mode_menu) {
    controls.append(makeButton("Exit", exitApp, "button secondary"));
    return;
  }
  controls.append(makeButton("Modes", () => {
    state.showingSettings = false;
    request("/api/mode-menu", {});
  }, "button secondary"));
  controls.append(makeButton(state.showingSettings ? "Back to table" : "Settings", () => {
    state.showingSettings = !state.showingSettings;
    render();
  }, "button secondary"));
  const undo = makeButton("Undo", () => {
    stopAiSchedule();
    request("/api/undo", {});
  }, "button secondary");
  undo.disabled = !game.can_undo || game.spectating;
  controls.append(undo);
  if (game.mode === "play") {
    controls.append(makeButton("New round", () => request("/api/new", {})));
  } else if (game.ready || game.advisor_stage !== "first_eight") {
    controls.append(makeButton("Set new advisor hand", () => {
      request("/api/advisor/reset", {});
    }, "button secondary"));
  }
  controls.append(makeButton("Exit", exitApp, "button secondary"));
}

function advisorCardForm({
  placeholder,
  submitLabel,
  onSubmit,
}) {
  const form = document.createElement("form");
  form.className = "advisor-card-form";
  const input = document.createElement("input");
  input.required = true;
  input.autocomplete = "off";
  input.placeholder = placeholder;
  const submit = makeButton(submitLabel, () => {});
  submit.type = "submit";
  form.append(input, submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    onSubmit(input.value);
  });
  return form;
}

function toggleGtCaller(player) {
  if (state.gtCallers.includes(player)) {
    state.gtCallers = state.gtCallers.filter((item) => item !== player);
  } else {
    state.gtCallers = [...state.gtCallers, player].sort();
  }
  render();
}

function renderSetup(game) {
  const setup = document.querySelector("#setup");
  if (game.show_mode_menu) {
    setup.classList.remove("hidden");
    setup.replaceChildren();
    const heading = document.createElement("h2");
    heading.textContent = "Choose mode";
    const modes = document.createElement("div");
    modes.className = "mode-options";
    modes.append(
      makeButton("Play", () => request("/api/mode", { mode: "play" }), "button mode-button"),
      makeButton("Advisor", () => request("/api/mode", { mode: "advisor" }), "button mode-button"),
    );
    setup.append(heading, modes);
    return;
  }
  if (game.ready) {
    setup.classList.add("hidden");
    return;
  }
  setup.classList.remove("hidden");
  setup.replaceChildren();
  if (game.mode === "play") {
    setup.innerHTML = "<h2>Ready to deal</h2><p>Play against the locally loaded model. The table includes Grand Tichu, directed passing, Mahjong wishes and Dragon gifts.</p>";
    setup.append(makeButton("Deal a visual round", () => request("/api/new", {})));
    return;
  }
  if (game.advisor_stage === "remaining_six") {
    const title = document.createElement("h2");
    title.textContent = "Grand Tichu";
    const recommendation = document.createElement("p");
    recommendation.className = "gt-recommendation";
    if (typeof game.gt_probability === "number") {
      const call = game.gt_probability >= 0.5;
      recommendation.textContent = `${call ? "Call Grand Tichu" : "Do not call Grand Tichu"} · ${Math.round(game.gt_probability * 100)}% call probability`;
      recommendation.classList.toggle("call", call);
    } else {
      recommendation.textContent = "No Grand Tichu model is loaded.";
    }
    const firstLabel = document.createElement("div");
    firstLabel.className = "setup-label";
    firstLabel.textContent = "First eight";
    const gtLabel = document.createElement("div");
    gtLabel.className = "setup-label";
    gtLabel.textContent = "Actual Grand Tichu callers";
    const callers = document.createElement("div");
    callers.className = "caller-options";
    [0, 1, 2, 3].forEach((player) => {
      callers.append(makeButton(
        player === game.seat ? `P${player} · You` : `P${player}`,
        () => toggleGtCaller(player),
        `button secondary ${state.gtCallers.includes(player) ? "active" : ""}`,
      ));
    });
    const remainingLabel = document.createElement("div");
    remainingLabel.className = "setup-label";
    remainingLabel.textContent = "Remaining six";
    setup.append(
      title,
      recommendation,
      firstLabel,
      cardsElement(game.first_eight || [], { className: "setup-cards" }),
      gtLabel,
      callers,
      remainingLabel,
      advisorCardForm({
        placeholder: "A gs 10 rgb m · 6 cards",
        submitLabel: "Continue to passing",
        onSubmit: (cards) => request("/api/advisor/remaining", {
          cards,
          gt_callers: state.gtCallers,
        }),
      }),
    );
    return;
  }

  setup.innerHTML = "<h2>First eight cards</h2>";
  const hint = document.createElement("p");
  hint.textContent = "Shortcuts: m Mahjong · d Dragon · w Dog · p Phoenix · 2 rgb · 2-7 r · J bomb";
  setup.append(
    hint,
    advisorCardForm({
      placeholder: "3g Jb 10r As m 2 rgb · 8 cards",
      submitLabel: "Check Grand Tichu",
      onSubmit: (cards) => request("/api/advisor/first-eight", { cards }),
    }),
  );
}

function renderSeat(seat) {
  const node = document.createElement("section");
  const position = seat.relation === "Left" ? "west" : seat.relation === "Right" ? "east" : "north";
  node.className = `seat ${position} ${seat.current ? "current" : ""} ${seat.finished ? "finished" : ""}`;
  const declaration = seat.tichu === 2 ? "GT" : seat.tichu === 1 ? "TICHU" : "";
  node.innerHTML = `<div class="seat-header">P${seat.player} · ${seat.relation}${declaration ? `<span class="declaration">${declaration}</span>` : ""}</div>
    <div class="seat-detail">${seat.finished ? `out #${seat.finish_position}` : `${seat.count} cards`}</div>`;
  const backs = document.createElement("div");
  backs.className = "card-backs";
  for (let i = 0; i < Math.min(seat.count, 9); i += 1) {
    const back = document.createElement("i");
    back.className = "card-back";
    backs.append(back);
  }
  node.append(backs);
  return node;
}

function toggleCard(index) {
  if (state.selected.includes(index)) state.selected = state.selected.filter((item) => item !== index);
  else state.selected = [...state.selected, index];
  render();
}

function clearSelection() {
  state.selected = [];
  render();
}

function resetPassSelection() {
  state.selected = [];
  state.passCards = [null, null, null];
  state.passSlot = 0;
}

function submitPass() {
  const cards = [...state.passCards];
  // Clear immediately instead of waiting for the AI to finish the exchange.
  // A passing choice must never become a selected play on the following turn.
  resetPassSelection();
  request("/api/pass", { cards });
}

function setPassingStyle(passingStyle) {
  state.settings.passingStyle = passingStyle;
  saveSettings();
  resetPassSelection();
  render();
}

function setShowAiSuggestions(showAiSuggestions) {
  state.settings.showAiSuggestions = showAiSuggestions;
  saveSettings();
  render();
}

function saveSettings() {
  try {
    window.localStorage.setItem("tichubot-settings", JSON.stringify(state.settings));
  } catch (_) {
    // The UI still works when a browser blocks local storage.
  }
}

function setAiRevealDelay(aiRevealDelayMs) {
  state.settings.aiRevealDelayMs = aiRevealDelayMs;
  saveSettings();
  render();
}

function setReplayRevealDelay(replayRevealDelayMs) {
  state.settings.replayRevealDelayMs = replayRevealDelayMs;
  saveSettings();
  render();
}

function delayLabel(delayMs) {
  if (delayMs === 0) return "Instant";
  if (delayMs < 1000) return `${delayMs / 1000} seconds`;
  return `${delayMs / 1000} second${delayMs === 1000 ? "" : "s"}`;
}

function delayDetail(delayMs) {
  if (delayMs === 0) return "Reveal the selected AI cards immediately.";
  return `Show each selected AI play after ${delayLabel(delayMs).toLowerCase()}.`;
}

function choosePassSlot(slot) {
  state.passSlot = slot;
  render();
}

function nextEmptyPassSlot(after) {
  for (let step = 1; step <= 3; step += 1) {
    const slot = (after + step) % 3;
    if (state.passCards[slot] === null) return slot;
  }
  return after;
}

function assignPassCard(index) {
  const target = state.passSlot;
  const existing = state.passCards.indexOf(index);
  if (existing !== -1 && existing !== target) {
    [state.passCards[existing], state.passCards[target]] = [
      state.passCards[target], state.passCards[existing],
    ];
  } else {
    state.passCards[target] = index;
  }
  state.passSlot = nextEmptyPassSlot(target);
  render();
}

function toggleOrderedPassCard(index) {
  let cards = state.passCards.filter((card) => card !== null);
  if (cards.includes(index)) cards = cards.filter((card) => card !== index);
  else if (cards.length < 3) cards.push(index);
  state.passCards = [...cards, ...Array(3 - cards.length).fill(null)];
  state.passSlot = cards.length < 3 ? cards.length : 2;
  render();
}

function renderTable(game) {
  const table = document.querySelector("#table");
  if (!game.ready) {
    table.classList.add("hidden");
    return;
  }
  table.classList.remove("hidden");
  table.replaceChildren();
  game.seats.filter((seat) => seat.player !== game.seat).forEach((seat) => table.append(renderSeat(seat)));

  const center = document.createElement("section");
  center.className = "table-center";
  const detail = game.table.length
    ? `${game.table_combo || "Play"} by P${game.table_leader} · ${game.passes} pass${game.passes === 1 ? "" : "es"}`
    : "New trick";
  center.innerHTML = `<div class="table-title">TABLE</div><div class="table-detail">${detail}${game.wish ? ` · wish: ${game.wish}` : ""}</div><div class="table-detail">${predictedValueText(game)}</div>`;
  center.append(cardsElement(game.table, { className: "table-cards" }));
  table.append(center);

  const hand = document.createElement("section");
  hand.className = "hand-area";
  hand.innerHTML = `<div class="hand-heading">${game.spectating ? `P${game.seat}'s replay hand` : "Your hand"} · ${game.hand.length} cards</div>`;
  const passing = !game.spectating
    && game.phase === "passing"
    && game.current_player === game.seat
    && (game.mode !== "advisor" || game.advisor_stage === "passing");
  const canSelect = game.human_turn || passing;
  hand.append(cardsElement(game.hand, {
    className: `hand-cards${game.hand.length >= 10 ? " dense-hand" : ""}`,
    selectable: canSelect,
    selectedIndices: passing ? state.passCards.filter((card) => card !== null) : state.selected,
    onCard: passing
      ? (state.settings.passingStyle === "ordered" ? toggleOrderedPassCard : assignPassCard)
      : toggleCard,
  }));
  table.append(hand);
  refreshSprites(table);
}

function renderSuggestions(game, container) {
  if (!state.settings.showAiSuggestions || !game.suggestions?.length) return;
  const row = document.createElement("div");
  row.className = "control-row";
  const title = document.createElement("span");
  title.className = "help";
  title.textContent = "AI suggestions:";
  row.append(title);
  const suggestions = document.createElement("div");
  suggestions.className = "suggestions";
  game.suggestions.forEach((suggestion) => {
    // Keep the face cards as disabled buttons, so the enclosing clickable
    // suggestion must be a non-button element (nested buttons are invalid
    // HTML and browsers handle them inconsistently).
    const button = document.createElement("div");
    button.className = "suggestion";
    button.tabIndex = 0;
    button.setAttribute("role", "button");
    button.title = cardNames(suggestion.cards);
    button.append(cardsElement(suggestion.cards));
    const label = document.createElement("span");
    label.textContent = suggestion.label + (suggestion.call_tichu ? " + Tichu" : "");
    const probability = document.createElement("small");
    probability.textContent = `${Math.round(suggestion.probability * 100)}%`;
    button.append(label, probability);
    const selectSuggestion = () => {
      state.selected = suggestion.indices;
      render();
    };
    button.addEventListener("click", selectSuggestion);
    button.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectSuggestion();
      }
    });
    suggestions.append(button);
  });
  row.append(suggestions);
  container.append(row);
}

function renderPassControls(game, container) {
  const row = document.createElement("div");
  row.className = "control-row";
  const info = document.createElement("div");
  info.className = "help";
  const ordered = state.settings.passingStyle === "ordered";
  info.textContent = ordered
    ? "Click three cards in order: left, partner, then right. Click a selected card again to remove it."
    : "Choose a recipient, then click a card in your hand. You must fill all three slots.";
  row.append(info);
  const slots = document.createElement("div");
  slots.className = "pass-slots";
  ["Left", "Partner", "Right"].forEach((label, index) => {
    const slot = document.createElement(ordered ? "div" : "button");
    if (!ordered) slot.type = "button";
    slot.className = `pass-slot ${!ordered && state.passSlot === index ? "active" : ""}`;
    const recipient = document.createElement("span");
    recipient.textContent = label;
    const chosen = document.createElement("strong");
    chosen.textContent = state.passCards[index] === null
      ? "Choose card"
      : game.hand.find((card) => card.index === state.passCards[index])?.name || "Choose card";
    slot.append(recipient, chosen);
    if (!ordered) slot.addEventListener("click", () => choosePassSlot(index));
    slots.append(slot);
  });
  row.append(slots);
  const submit = makeButton("Submit passing cards", submitPass, "button");
  submit.disabled = state.passCards.some((card) => card === null);
  row.append(submit);
  row.append(makeButton("Clear passes", () => { resetPassSelection(); render(); }, "button secondary"));
  container.append(row);
  if (game.pass_suggestion?.length) {
    const suggestion = document.createElement("div");
    suggestion.className = "control-row";
    suggestion.innerHTML = "<span class=\"help\">Passing model:</span>";
    const use = makeButton(`Use ${cardNames(game.pass_suggestion)}`, () => {
      state.passCards = game.pass_suggestion.map((card) => card.index);
      state.passSlot = 0;
      render();
    }, "button secondary");
    suggestion.append(use);
    container.append(suggestion);
  }
}

function renderSettings(game) {
  const settings = document.querySelector("#settings");
  if (!state.showingSettings) {
    settings.classList.add("hidden");
    return false;
  }
  settings.classList.remove("hidden");
  settings.replaceChildren();
  const heading = document.createElement("h2");
  heading.textContent = "Settings";
  const description = document.createElement("p");
  description.textContent = "These preferences stay in this browser and do not alter the Tichu rules.";
  const section = document.createElement("section");
  section.className = "settings-group";
  const title = document.createElement("h3");
  title.textContent = "Passing controls";
  section.append(title);
  [
    ["ordered", "Ordered card clicks", "Click cards in sequence: Left → Partner → Right."],
    ["slots", "Recipient slots", "Choose Left, Partner, or Right first, then click its card."],
  ].forEach(([value, label, detail]) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = `settings-option ${state.settings.passingStyle === value ? "active" : ""}`;
    const optionTitle = document.createElement("strong");
    optionTitle.textContent = label;
    const optionDetail = document.createElement("span");
    optionDetail.textContent = detail;
    option.append(optionTitle, optionDetail);
    option.addEventListener("click", () => setPassingStyle(value));
    section.append(option);
  });
  const suggestionsSection = document.createElement("section");
  suggestionsSection.className = "settings-group";
  const suggestionsTitle = document.createElement("h3");
  suggestionsTitle.textContent = "AI suggestions";
  suggestionsSection.append(suggestionsTitle);
  [
    [true, "Show suggestions", "Display the model's ranked plays when it is your turn."],
    [false, "Hide suggestions", "Keep the model available for opponents and replay, without showing move hints."],
  ].forEach(([value, label, detail]) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = `settings-option ${state.settings.showAiSuggestions === value ? "active" : ""}`;
    const optionTitle = document.createElement("strong");
    optionTitle.textContent = label;
    const optionDetail = document.createElement("span");
    optionDetail.textContent = detail;
    option.append(optionTitle, optionDetail);
    option.addEventListener("click", () => setShowAiSuggestions(value));
    suggestionsSection.append(option);
  });
  const revealSection = document.createElement("section");
  revealSection.className = "settings-group";
  const revealTitle = document.createElement("h3");
  revealTitle.textContent = "Opponent card reveal delay";
  const revealHelp = document.createElement("p");
  revealHelp.className = "settings-help";
  revealHelp.textContent = "This pauses only opponents after they have chosen a normal play. Your partner reveals immediately. It never slows model decisions, passing, Grand Tichu, wishes, or Dragon gifts.";
  revealSection.append(revealTitle, revealHelp);
  [0, 250, 500, 1000, 1500].forEach((delayMs) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = `settings-option ${state.settings.aiRevealDelayMs === delayMs ? "active" : ""}`;
    const optionTitle = document.createElement("strong");
    optionTitle.textContent = delayMs === 0 ? "Instant (default)" : delayLabel(delayMs);
    const optionDetail = document.createElement("span");
    optionDetail.textContent = delayDetail(delayMs);
    option.append(optionTitle, optionDetail);
    option.addEventListener("click", () => setAiRevealDelay(delayMs));
    revealSection.append(option);
  });
  const replaySection = document.createElement("section");
  replaySection.className = "settings-group";
  const replayTitle = document.createElement("h3");
  replayTitle.textContent = "AI replay reveal delay";
  const replayHelp = document.createElement("p");
  replayHelp.className = "settings-help";
  replayHelp.textContent = "When spectating a replay, this pace applies to all four AI seats.";
  replaySection.append(replayTitle, replayHelp);
  [0, 250, 500, 1000, 1500].forEach((delayMs) => {
    const option = document.createElement("button");
    option.type = "button";
    option.className = `settings-option ${state.settings.replayRevealDelayMs === delayMs ? "active" : ""}`;
    const optionTitle = document.createElement("strong");
    optionTitle.textContent = delayMs === 500 ? `${delayLabel(delayMs)} (default)` : delayLabel(delayMs);
    const optionDetail = document.createElement("span");
    optionDetail.textContent = delayDetail(delayMs);
    option.append(optionTitle, optionDetail);
    option.addEventListener("click", () => setReplayRevealDelay(delayMs));
    replaySection.append(option);
  });
  settings.append(heading, description, section, suggestionsSection, revealSection, replaySection);
  return true;
}

function renderAdvisorOpponent(game, container) {
  if (
    game.mode !== "advisor"
    || game.current_player === game.seat
    || game.round_over
    || game.phoenix_choices?.length
    || Number.isInteger(game.dragon_give_pending)
  ) return;
  const form = document.createElement("form");
  form.className = "opponent-form";
  const input = document.createElement("input");
  input.placeholder = `Record P${game.current_player}: cards, or leave empty for pass`;
  const tichu = document.createElement("label");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  tichu.append(checkbox, " called Tichu");
  const submit = makeButton(`Record P${game.current_player}`, () => {});
  submit.type = "submit";
  form.append(input, tichu, submit);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    request("/api/advisor/opponent", { player: game.current_player, cards: input.value, call_tichu: checkbox.checked });
  });
  container.append(form);
}

function timelineElement() {
  let timeline = document.querySelector("#history");
  if (timeline) return timeline;
  timeline = document.createElement("section");
  timeline.id = "history";
  timeline.className = "timeline";
  const controls = document.querySelector("#controls");
  // Keep the action controls before history in the document order.  CSS moves
  // history into a side rail on wide screens; narrow screens retain this
  // practical order so the play button is never below a long event list.
  controls.parentNode.insertBefore(timeline, controls.nextSibling);
  return timeline;
}

function renderTimeline(game) {
  const timeline = timelineElement();
  if (!game.ready || state.showingSettings) {
    timeline.classList.add("hidden");
    return;
  }
  timeline.classList.remove("hidden");
  timeline.replaceChildren();
  const heading = document.createElement("h2");
  heading.textContent = "Play history";
  const history = document.createElement("ol");
  history.setAttribute("aria-live", "polite");
  (game.history || []).forEach((event) => {
    const item = document.createElement("li");
    item.dataset.eventType = event.type || "event";
    item.textContent = event.text || "Game event.";
    if (event.cards?.length && event.type !== "pass_submission") {
      const cards = document.createElement("span");
      cards.textContent = ` — ${cardNames(event.cards)}`;
      item.append(cards);
    }
    history.append(item);
  });
  timeline.append(heading, history);
}

function replayCommand(path, { playing = false } = {}) {
  state.replayPlaying = playing;
  stopAiSchedule();
  return request(path, {});
}

function renderReplayControls(game, controls) {
  const row = document.createElement("div");
  row.className = "control-row replay-controls";

  const back = makeButton("Back", () => replayCommand("/api/replay/back"), "button secondary");
  back.disabled = !game.replay_can_back;

  const play = makeButton(
    state.replayPlaying ? "Pause" : "Play",
    () => {
      state.replayPlaying = !state.replayPlaying;
      render();
    },
    "button secondary",
  );
  play.disabled = !game.replay_can_forward;

  const forward = makeButton("Forward", () => replayCommand("/api/replay/forward"), "button secondary");
  forward.disabled = !game.replay_can_forward;

  const end = makeButton("End", () => replayCommand("/api/replay/end"), "button secondary");
  end.disabled = !game.replay_can_forward;

  const position = document.createElement("span");
  position.className = "help replay-position";
  position.textContent = `Step ${game.replay_cursor}`;
  row.append(back, play, forward, end, position);
  controls.append(row);
}

function renderControls(game) {
  const controls = document.querySelector("#controls");
  if (!game.ready) {
    controls.classList.add("hidden");
    return;
  }
  controls.classList.remove("hidden");
  controls.replaceChildren();
  const title = document.createElement("h2");
  title.textContent = game.round_over
    ? game.spectating ? "AI replay complete" : "Round over"
    : game.spectating ? "Spectating AI replay" : `Phase: ${game.phase.replace("_", " ")}`;
  controls.append(title);
  if (game.spectating) renderReplayControls(game, controls);
  if (game.round_over) {
    const score = document.createElement("p");
    score.className = "score";
    score.textContent = scoreText(game);
    controls.append(score);
    if (game.can_replay) {
      controls.append(makeButton(
        game.spectating ? "Replay this deal again" : "Replay this deal with AI",
        () => replayCommand("/api/replay", { playing: true }),
      ));
    }
    return;
  }
  if (game.spectating) {
    const watch = document.createElement("p");
    watch.className = "help";
    watch.textContent = state.replayPlaying
      ? game.ai_pending
        ? `P${aiPlayer(game)} has selected a play; revealing it shortly…`
        : "The AI is resolving the replay…"
      : "Replay paused.";
    controls.append(watch);
    return;
  }
  if (game.phase === "grand_tichu" && game.current_player === game.seat) {
    const row = document.createElement("div");
    row.className = "control-row";
    row.append(document.createTextNode("Call Grand Tichu with the first eight cards?"));
    row.append(makeButton("Call Grand Tichu", () => request("/api/grand-tichu", { declare: true }), "button danger"));
    row.append(makeButton("Do not call", () => request("/api/grand-tichu", { declare: false }), "button secondary"));
    controls.append(row);
  } else if (game.mode === "advisor" && game.advisor_stage === "received") {
    const label = document.createElement("div");
    label.className = "help";
    label.textContent = "Received from left, partner, right";
    controls.append(
      label,
      advisorCardForm({
        placeholder: "2r 5g Dragon · left, partner, right",
        submitLabel: "Complete passing",
        onSubmit: (cards) => request("/api/advisor/received", { cards }),
      }),
    );
  } else if (game.mode === "advisor" && game.advisor_stage === "mahjong_holder") {
    const row = document.createElement("div");
    row.className = "control-row";
    row.append(document.createTextNode("Who holds the Mahjong?"));
    game.seats
      .filter((seat) => seat.player !== game.seat)
      .forEach((seat) => row.append(makeButton(
        `P${seat.player} · ${seat.relation}`,
        () => request("/api/advisor/mahjong-holder", { holder: seat.player }),
        "button secondary",
      )));
    controls.append(row);
  } else if (game.phase === "passing" && game.current_player === game.seat) {
    renderPassControls(game, controls);
  } else if (game.phoenix_choices?.length) {
    const row = document.createElement("div");
    row.className = "control-row";
    row.append(document.createTextNode("Choose the Phoenix value:"));
    game.phoenix_choices.forEach((choice) => {
      row.append(makeButton(
        choice.label,
        () => request("/api/phoenix", { phoenix_as: choice.value }),
        "button secondary",
      ));
    });
    row.append(makeButton(
      "Cancel",
      () => request("/api/phoenix/cancel", {}),
      "button secondary",
    ));
    controls.append(row);
  } else if (game.waiting_for_wish) {
    const row = document.createElement("div");
    row.className = "control-row";
    row.append(document.createTextNode("Mahjong wish:"));
    [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"].forEach((label, index) => {
      row.append(makeButton(String(label), () => request("/api/wish", { wish: index + 2 }), "button secondary"));
    });
    row.append(makeButton("No wish", () => request("/api/wish", { wish: null }), "button secondary"));
    controls.append(row);
  } else if (
    game.dragon_give_pending === game.seat
    || (game.mode === "advisor" && Number.isInteger(game.dragon_give_pending))
  ) {
    // In advisor mode an opponent's real-world Dragon gift is recorded here too.
    const giver = game.dragon_give_pending;
    const row = document.createElement("div");
    row.className = "control-row";
    row.append(document.createTextNode(
      giver === game.seat ? "Give the Dragon trick to:" : `P${giver} gave the Dragon trick to:`,
    ));
    game.seats.filter((seat) => seat.player % 2 !== giver % 2).forEach((seat) => {
      row.append(makeButton(`P${seat.player}`, () => request("/api/dragon", { recipient: seat.player }), "button secondary"));
    });
    controls.append(row);
  } else if (game.human_turn) {
    const row = document.createElement("div");
    row.className = "control-row";
    const play = makeButton(`Play selected (${state.selected.length})`, () => request("/api/play", { cards: state.selected, call_tichu: false }));
    const tichu = makeButton("Play + call Tichu", () => request("/api/play", { cards: state.selected, call_tichu: true }), "button danger");
    row.append(play, tichu, makeButton("Pass", () => request("/api/play", { cards: [], call_tichu: false }), "button secondary"), makeButton("Clear", clearSelection, "button secondary"));
    controls.append(row);
    renderSuggestions(game, controls);
  } else {
    const wait = document.createElement("p");
    wait.className = "help";
    const pendingPlayer = aiPlayer(game);
    const pendingPlayerLabel = Number.isInteger(pendingPlayer)
      ? `P${pendingPlayer}`
      : "An AI player";
    const delay = aiRevealDelay(game);
    wait.textContent = game.mode === "advisor"
      ? `Record P${game.current_player}'s real-world play below.`
      : game.ai_pending
        ? delay > 0
          ? `${pendingPlayerLabel} has chosen a play; revealing it shortly…`
          : `${pendingPlayerLabel} is playing…`
        : "Waiting for the next play…";
    controls.append(wait);
  }
  renderAdvisorOpponent(game, controls);
}

function render() {
  const game = state.game;
  if (!game) return;
  renderHeader(game);
  const message = document.querySelector("#message");
  message.classList.toggle("error", Boolean(state.error));
  message.textContent = state.error || game.message || "";
  if (game.show_mode_menu) {
    state.showingSettings = false;
    document.querySelector("#settings").classList.add("hidden");
    renderSetup(game);
    renderTable(game);
    renderControls(game);
    renderTimeline(game);
    stopAiSchedule();
    return;
  }
  const settingsOpen = renderSettings(game);
  if (settingsOpen) {
    document.querySelector("#setup").classList.add("hidden");
    document.querySelector("#table").classList.add("hidden");
    document.querySelector("#controls").classList.add("hidden");
    renderTimeline(game);
    stopAiSchedule();
    return;
  }
  renderSetup(game);
  renderTable(game);
  renderControls(game);
  renderTimeline(game);
  requestAnimationFrame(refreshSprites);
  scheduleAiStep(game);
}

window.addEventListener("resize", () => refreshSprites());
request("/api/state");
