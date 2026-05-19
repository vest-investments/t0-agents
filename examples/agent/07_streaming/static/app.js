// Minimal browser client for the t0-agents streaming WS.
//
// Every AgentEvent `type` maps to a small rendering routine below.  No
// framework — vanilla DOM + WebSocket — so the file fits on one screen
// and the wire protocol is visible end to end.
//
// Events we render:
//   text_delta          → append streamed text to the current assistant msg
//   text_end            → seal the text block (no visible effect here)
//   tool_call           → inline "Calling <name>…" spinner
//   tool_result         → replace the spinner with the result string
//   thought_delta       → muted "thinking" bubble
//   thought             → seal the thought bubble
//   turn_end            → visual separator between turns
//   done                → stop the spinner, capture episode_id
//   error               → red bubble with the error message
//   elicitation_request → open the approval modal; send back
//                          elicitation_response on accept/decline
//   ws_error            → protocol-level error; same rendering as error

const logEl = document.getElementById("chat-log");
const statusEl = document.getElementById("status");
const composer = document.getElementById("composer");
const input = document.getElementById("message");
const sendBtn = composer.querySelector("button");
const modal = document.getElementById("elicit-modal");
const modalTitle = document.getElementById("elicit-title");
const modalMessage = document.getElementById("elicit-message");
const modalFields = document.getElementById("elicit-fields");

let ws = null;
let episodeId = null;
let currentAssistant = null;
const toolBubbles = new Map();      // id -> DOM node
const thoughtBubbles = new Map();

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/chat/ws`);
  ws.onopen = () => { statusEl.textContent = "Connected."; sendBtn.disabled = false; };
  ws.onclose = () => {
    statusEl.textContent = "Disconnected — reconnecting…";
    sendBtn.disabled = true;
    setTimeout(connect, 1000);
  };
  ws.onerror = (err) => {
    statusEl.textContent = "WebSocket error; see console.";
    console.error(err);
  };
  ws.onmessage = (msg) => {
    const frame = JSON.parse(msg.data);
    handleFrame(frame);
  };
}

function handleFrame(frame) {
  switch (frame.type) {
    case "text_delta":
      appendToAssistant(frame.delta);
      break;
    case "text_end":
      // Let subsequent turns open a fresh assistant bubble.
      currentAssistant = null;
      break;
    case "thought_delta":
      appendToThought(frame.id, frame.delta);
      break;
    case "thought":
      // Seal the thought bubble so new thoughts get a new one.
      thoughtBubbles.delete(frame.id);
      break;
    case "tool_call":
      showToolSpinner(frame.id, frame.name);
      break;
    case "tool_result":
      finishToolBubble(frame.id, frame.result, frame.is_error);
      break;
    case "turn_end":
      // Visual separator is optional — we just reset bubble tracking.
      currentAssistant = null;
      break;
    case "done":
      episodeId = frame.result.episode_id;
      statusEl.textContent = `Ready — episode ${episodeId || "?"}, ` +
        `${frame.result.turns} turn(s).`;
      sendBtn.disabled = false;
      break;
    case "error":
    case "ws_error":
      renderErrorBubble(frame.message);
      sendBtn.disabled = false;
      break;
    case "elicitation_request":
      openElicitModal(frame);
      break;
    default:
      console.warn("unknown frame", frame);
  }
}

function appendToAssistant(delta) {
  if (!currentAssistant) {
    currentAssistant = document.createElement("div");
    currentAssistant.className = "msg assistant";
    logEl.appendChild(currentAssistant);
  }
  currentAssistant.textContent += delta;
  scrollToEnd();
}

function appendToThought(id, delta) {
  let el = thoughtBubbles.get(id);
  if (!el) {
    el = document.createElement("div");
    el.className = "msg tool";
    el.textContent = "(thinking) ";
    logEl.appendChild(el);
    thoughtBubbles.set(id, el);
  }
  el.textContent += delta;
  scrollToEnd();
}

function showToolSpinner(id, name) {
  const el = document.createElement("div");
  el.className = "msg tool";
  el.innerHTML = `<span class="spinner"></span>Calling <b>${escape(name)}</b>…`;
  logEl.appendChild(el);
  toolBubbles.set(id, el);
  scrollToEnd();
}

function finishToolBubble(id, result, isError) {
  const el = toolBubbles.get(id);
  if (!el) return;
  const prefix = isError ? "⚠️  " : "↳ ";
  el.innerHTML = `${prefix}<code>${escape(result)}</code>`;
  if (isError) el.classList.add("error");
  toolBubbles.delete(id);
}

function renderErrorBubble(message) {
  const el = document.createElement("div");
  el.className = "msg error";
  el.textContent = `⚠️  ${message}`;
  logEl.appendChild(el);
  scrollToEnd();
}

function openElicitModal(frame) {
  modalTitle.textContent = frame.response_type
    ? "Provide details"
    : "Approval required";
  modalMessage.textContent = frame.message;
  modalFields.innerHTML = "";

  const schema = frame.schema_ ?? frame.schema ?? null;
  const props = schema?.properties ?? {};
  const inputs = {};
  for (const [name, spec] of Object.entries(props)) {
    const label = document.createElement("label");
    label.textContent = `${name} (${spec.type ?? "string"})`;
    modalFields.appendChild(label);
    let inp;
    if (spec.enum) {
      inp = document.createElement("select");
      for (const opt of spec.enum) {
        const o = document.createElement("option");
        o.value = o.textContent = opt;
        inp.appendChild(o);
      }
    } else {
      inp = document.createElement("input");
      inp.type = spec.type === "integer" || spec.type === "number"
        ? "number" : "text";
    }
    inp.name = name;
    if (spec.default !== undefined) inp.value = spec.default;
    modalFields.appendChild(inp);
    inputs[name] = inp;
  }

  modal.showModal();

  modal.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.onclick = () => {
      const action = btn.dataset.action;
      const payload = { action };
      if (action === "accept") {
        const content = {};
        for (const [name, inp] of Object.entries(inputs)) {
          const spec = props[name];
          content[name] = (spec?.type === "integer")
            ? parseInt(inp.value, 10)
            : (spec?.type === "number")
              ? parseFloat(inp.value)
              : inp.value;
        }
        // Scalar wrapping: if no structured fields, FastMCP expects
        // content to be an empty object for binary approval.
        payload.content = Object.keys(content).length > 0 ? content : {};
      }
      ws.send(JSON.stringify({
        type: "elicitation_response",
        elicitation_id: frame.elicitation_id,
        payload,
      }));
      modal.close();
    };
  });
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  const user = document.createElement("div");
  user.className = "msg user";
  user.textContent = text;
  logEl.appendChild(user);
  currentAssistant = null;
  sendBtn.disabled = true;
  statusEl.textContent = "Streaming…";
  ws.send(JSON.stringify({
    type: "user_message",
    message: text,
    episode_id: episodeId,
  }));
  input.value = "";
  scrollToEnd();
});

function scrollToEnd() {
  logEl.scrollTop = logEl.scrollHeight;
}

function escape(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

connect();
