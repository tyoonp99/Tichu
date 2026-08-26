const params = new URLSearchParams(location.search);
let roomId = params.get("room");
let seat = Number(params.get("seat"));
let token = params.get("token");
let poller;
let selectedCards = new Set();
let lastState;
let renderedControlKey;

const byId = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "요청에 실패했습니다.");
  return data;
}

async function createRoom(mode) {
  const room = await api("/api/rooms", { method: "POST", body: JSON.stringify({ mode }) });
  if (room.friend_token) sessionStorage.setItem(`tichu-friend-token-${room.room_id}`, room.friend_token);
  location.search = new URLSearchParams({ room: room.room_id, seat: "0", token: room.host_token });
}

function stateUrl(suffix = "") {
  return `/api/rooms/${roomId}/state/${seat}${suffix}?token=${encodeURIComponent(token)}`;
}

async function join() {
  if (!roomId || !token || ![0, 2].includes(seat)) return;
  await api(`/api/rooms/${roomId}/join?token=${encodeURIComponent(token)}`, { method: "POST", body: JSON.stringify({ seat }) });
  byId("lobby").hidden = true;
  byId("game").hidden = false;
  byId("room").textContent = `방 ${roomId} · 당신은 ${seat}번 · 파트너는 ${(seat + 2) % 4}번`;
  if (seat === 0 && sessionStorage.getItem(`tichu-friend-token-${roomId}`)) {
    const friendUrl = new URL(location.href);
    friendUrl.searchParams.set("seat", "2");
    friendUrl.searchParams.set("token", sessionStorage.getItem(`tichu-friend-token-${roomId}`) || "");
    byId("invite").textContent = `친구 초대 링크: ${friendUrl.href}`;
  }
  await refresh();
  poller = setInterval(refresh, 250);
}

function addButtons(container, actions, onClick, replace = true) {
  if (replace) container.innerHTML = "";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.textContent = action.label ?? action;
    button.onclick = () => onClick(action.id ?? action);
    container.append(button);
  });
}

function sameCards(left, right) {
  return left.length === right.length && [...left].sort().join("|") === [...right].sort().join("|");
}

function controlKey(state) {
  // Action ids deliberately change on each server response.  They must not
  // cause a full hand/button repaint every 250 ms while the player is clicking.
  return JSON.stringify({
    hand: state.hand,
    table: state.table_cards,
    waitingPlayer: state.waiting_player,
    turnKind: state.turn_kind,
    prompt: state.prompt,
    actions: state.legal_actions.map(({ label, cards, kind }) => ({ label, cards, kind })),
  });
}

function seatName(number) {
  if (number === seat) return `나 · ${number}번`;
  if (number === (seat + 2) % 4) return `파트너 · ${number}번`;
  return `상대 AI · ${number}번`;
}

function renderSeats(state) {
  [0, 1, 2, 3].forEach((number) => {
    const element = byId(`seat-${number}`);
    const isMe = number === seat;
    const isPartner = number === (seat + 2) % 4;
    element.className = `seat ${number === 0 ? "bottom" : number === 1 ? "right" : number === 2 ? "top" : "left"}${state.current_player === number ? " active" : ""}${isMe ? " me" : ""}`;
    element.innerHTML = `<strong>${seatName(number)}</strong><span>${state.hand_sizes[number]}장${isMe ? " · 내 손패" : isPartner ? " · 같은 팀" : ""}</span>`;
  });
}

function renderHand(state) {
  const hand = byId("hand");
  hand.innerHTML = "";
  const available = new Set(state.hand);
  selectedCards.forEach((card) => { if (!available.has(card)) selectedCards.delete(card); });
  state.hand.forEach((card) => {
    const button = document.createElement("button");
    button.className = `card${selectedCards.has(card) ? " selected" : ""}`;
    button.textContent = card;
    button.disabled = state.waiting_player !== seat;
    button.onclick = () => {
      selectedCards.has(card) ? selectedCards.delete(card) : selectedCards.add(card);
      renderHand(lastState);
      renderActions(lastState);
    };
    hand.append(button);
  });
}

function renderActions(state) {
  const selected = [...selectedCards];
  const playActions = state.legal_actions.filter((action) => action.cards.length > 0);
  const selectedActions = playActions.filter((action) => sameCards(action.cards, selected));
  const passes = state.legal_actions.filter((action) => action.kind === "PassAction" || action.kind === "PassBombAction");
  const controls = byId("actions");
  controls.innerHTML = "";
  const selection = byId("selection");
  selection.textContent = selected.length ? `선택: ${selected.join(" ")}` : "카드를 선택하세요. 싱글은 카드 한 장을 고른 뒤 바로 낼 수 있습니다.";

  if (selectedActions.length) {
    addButtons(controls, selectedActions.map((action) => ({ id: action.id, label: `내기 · ${action.label}` })), submitAction);
  } else {
    const playButton = document.createElement("button");
    playButton.className = "play-button";
    playButton.disabled = true;
    playButton.textContent = "내기";
    controls.append(playButton);
  }
  if (selected.length && !selectedActions.length) {
    const hint = document.createElement("span");
    hint.className = "invalid-selection";
    hint.textContent = "이 선택으로 낼 수 있는 합법 조합이 없습니다.";
    controls.append(hint);
  }
  if (selected.length) {
    const clearButton = document.createElement("button");
    clearButton.className = "clear-button";
    clearButton.textContent = "선택 초기화";
    clearButton.onclick = () => {
      selectedCards.clear();
      renderHand(lastState);
      renderActions(lastState);
    };
    controls.append(clearButton);
  }
  addButtons(controls, passes, submitAction, false);

  // This panel is sent only to the person whose private hand can play one.
  // It provides a one-click response during the shared three-second window.
  const bombs = playActions.filter((action) => action.kind === "SquareBomb" || action.kind === "StraightBomb");
  const bombPanel = byId("bomb-actions");
  bombPanel.hidden = state.turn_kind !== "PassBombAction" || bombs.length === 0;
  addButtons(byId("bomb-buttons"), bombs, submitAction);

}

async function submitAction(id) {
  try {
    render(await api(stateUrl("/action"), {
      method: "POST", body: JSON.stringify({ action_id: id }),
    }));
  } catch (error) { alert(error.message); }
}

async function submitPrompt(choice) {
  try {
    render(await api(stateUrl("/prompt"), {
      method: "POST", body: JSON.stringify({ choice }),
    }));
  } catch (error) { alert(error.message); }
}

function render(state) {
  lastState = state;
  renderSeats(state);
  byId("table").textContent = state.table || "";
  byId("table-cards").innerHTML = state.table_cards.length
    ? state.table_cards.map((card) => `<span class="table-card">${card}</span>`).join("")
    : "비어 있음";
  byId("table-player").textContent = state.table_player === null
    ? ""
    : `플레이어 ${state.table_player}번이 낸 카드`;
  byId("table-points").textContent = state.table_player === null
    ? ""
    : `이번 트릭: ${state.table_points}점`;
  const wish = byId("wish");
  wish.hidden = !state.wish;
  wish.textContent = state.wish ? `마작 소원: ${state.wish}` : "";
  byId("status").textContent = state.terminal_points
    ? `라운드 종료 · 인간 팀 ${state.terminal_points[0]} : AI 팀 ${state.terminal_points[1]}`
    : state.reaction_remaining !== null ? `다음 행동까지 ${state.reaction_remaining.toFixed(1)}초`
    : state.animation_remaining > 0 ? "카드를 테이블에 올리는 중…"
    : state.waiting_player === seat ? "당신의 차례입니다. 카드를 선택해 내세요."
    : "진행 중입니다.";
  const reactionTimer = byId("reaction-timer");
  reactionTimer.hidden = state.reaction_remaining === null;
  if (state.reaction_remaining !== null) {
    byId("reaction-label").textContent = `다음 행동 전 반응 시간 · ${state.reaction_remaining.toFixed(1)}초`;
    byId("reaction-fill").style.width = `${Math.max(0, state.reaction_remaining / 3) * 100}%`;
  }
  const banner = byId("event-banner");
  if (state.last_event && state.animation_remaining > 0) {
    banner.hidden = false;
    banner.textContent = `플레이어 ${state.last_event.actor}: ${state.last_event.label}`;
  } else {
    banner.hidden = true;
  }
  byId("log").innerHTML = state.log.map((line) => `<li>${line}</li>`).join("");
  const nextControlKey = controlKey(state);
  if (nextControlKey !== renderedControlKey) {
    renderedControlKey = nextControlKey;
    renderHand(state);
    renderActions(state);
  }
  const prompt = byId("prompt");
  prompt.innerHTML = "";
  if (state.prompt) {
    const title = document.createElement("h2");
    title.textContent = state.prompt.kind === "wish" ? "마작 소원을 고르세요" : "드래곤 트릭을 줄 상대를 고르세요";
    prompt.append(title);
    if (state.prompt.kind === "dragon") {
      const points = document.createElement("p");
      points.className = "dragon-points";
      points.textContent = `상대에게 줄 트릭 점수: ${state.prompt.points}점`;
      prompt.append(points);
    }
    addButtons(prompt, state.prompt.choices.map((choice) => ({ id: choice, label: String(choice) })), submitPrompt);
  }
}

async function refresh() {
  if (!roomId) return;
  try { render(await api(stateUrl())); }
  catch (error) { console.warn(error.message); }
}

byId("create-single").onclick = () => createRoom("single").catch((error) => alert(error.message));
byId("create-friends").onclick = () => createRoom("friends").catch((error) => alert(error.message));
join().catch((error) => alert(error.message));
