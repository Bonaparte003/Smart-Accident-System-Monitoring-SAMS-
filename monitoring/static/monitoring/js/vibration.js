import { alertForLabel, stopAlerts } from "./sounds.js?v=uploadlive1";

const state = {
  mode: "live",
};

let livePoll = null;

export function initVibrationLab() {
  document.getElementById("hw-buzzer-enabled").addEventListener("change", async (e) => {
    await fetch("/api/hardware/buzzer/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: e.target.checked }),
    });
  });

  document.getElementById("hw-buzzer-auto").addEventListener("change", async (e) => {
    await fetch("/api/hardware/buzzer/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto: e.target.checked }),
    });
  });

  document.getElementById("hw-leds-enabled").addEventListener("change", async (e) => {
    await fetch("/api/hardware/leds/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: e.target.checked }),
    });
  });

  document.getElementById("hw-leds-auto").addEventListener("change", async (e) => {
    await fetch("/api/hardware/leds/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto: e.target.checked }),
    });
  });

  document.getElementById("hw-buzz-off").addEventListener("click", () => setBuzzer("off"));
  document.getElementById("hw-buzz-caution").addEventListener("click", () => setBuzzer("caution"));
  document.getElementById("hw-buzz-danger").addEventListener("click", () => setBuzzer("danger"));
  document.getElementById("hw-buzz-test").addEventListener("click", testBuzzer);

  document.getElementById("hw-led-green").addEventListener("click", () => setLed("green"));
  document.getElementById("hw-led-yellow").addEventListener("click", () => setLed("yellow"));
  document.getElementById("hw-led-red").addEventListener("click", () => setLed("red"));
  document.getElementById("hw-led-off").addEventListener("click", () => setLed("off"));

  connectLive();
}

async function setBuzzer(buzzerState) {
  if (buzzerState === "off") stopAlerts();
  await fetch("/api/hardware/buzzer/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state: buzzerState }),
  });
}

async function testBuzzer() {
  const statusEl = document.getElementById("hw-buzz-test-status");
  const btn = document.getElementById("hw-buzz-test");
  const show = (msg, ok = true) => {
    if (!statusEl) return;
    statusEl.hidden = false;
    statusEl.textContent = msg;
    statusEl.style.color = ok ? "#0b6e4f" : "#e03121";
  };

  btn.disabled = true;
  show("Buzzing D8…");
  try {
    const res = await fetch("/api/hardware/buzzer/test/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seconds: 1.2 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    show("Hardware B1 sent — buzzer should beep on D8.");
  } catch (err) {
    show(String(err.message || err), false);
  } finally {
    btn.disabled = false;
  }
}

async function setLed(color) {
  const res = await fetch("/api/hardware/leds/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color }),
  });
  const data = await res.json();
  if (data.leds) updateLedUi(data.leds.color);
}

async function connectLive() {
  const hint = document.getElementById("vib-mode-hint");
  try {
    const res = await fetch("/api/hardware/mode/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "live" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    updateHwPill(data.hardware);
    if (data.leds) updateLedUi(data.leds.color);
    if (hint) hint.textContent = "Streaming real MPU6050 windows into the vibration model.";
    window.dispatchEvent(new CustomEvent("sams:hardware-mode", { detail: data.hardware }));
  } catch (err) {
    console.error(err);
    updateHwPill({ mode: "live", source: "serial", last_error: String(err.message || err) });
    if (hint) {
      hint.textContent = `Live MPU6050 not connected. Set SAMS_HW_SERIAL_PORT and reconnect the board. ${err.message || err}`;
    }
  }

  clearInterval(livePoll);
  livePoll = setInterval(pollLive, 400);
  setTimeout(pollLive, 200);
}

function updateHwPill(hw) {
  const pill = document.getElementById("hw-status-pill");
  if (!pill || !hw) return;
  const bits = ["live", "serial"];
  if (hw.connected) bits.push("connected");
  else bits.push("disconnected");
  if (hw.window_ready) bits.push("ready");
  else if (hw.window_size) bits.push(`buf ${hw.buffer_filled || 0}/${hw.window_size}`);
  if (hw.last_error) bits.push("err");
  pill.textContent = bits.join(" · ");
  pill.classList.toggle("live", !!hw.connected);
  pill.title = hw.last_error || hw.serial_port || "";
}

function updateLedUi(color) {
  const map = {
    green: "led-green",
    yellow: "led-yellow",
    red: "led-red",
  };
  Object.values(map).forEach((id) => document.getElementById(id)?.classList.remove("on"));
  if (color && map[color]) document.getElementById(map[color])?.classList.add("on");
  const label = document.getElementById("led-label");
  if (label) label.textContent = color && color !== "off" ? `LED ${color}` : "LEDs off";
}

async function pollLive() {
  try {
    const res = await fetch("/api/vibration/live/");
    const data = await res.json();
    if (data.hardware) updateHwPill(data.hardware);
    if (!data.ready) return;
    applyResult(data);
  } catch (err) {
    console.error(err);
  }
}

function applyResult(data) {
  drawWave(data.series);
  document.getElementById("vib-label").textContent = data.label;
  document.getElementById("vib-conf").textContent = `${Math.round(data.confidence * 100)}%`;
  document.getElementById("vib-xrms").textContent = data.feature_snapshot?.x_rms?.toFixed?.(3) ?? "—";
  document.getElementById("vib-yrms").textContent = data.feature_snapshot?.y_rms?.toFixed?.(3) ?? "—";
  const n = (data.probabilities?.NORMAL || 0) * 100;
  const a = (data.probabilities?.ABNORMAL || 0) * 100;
  document.getElementById("vib-bar-n").style.width = `${n}%`;
  document.getElementById("vib-bar-a").style.width = `${a}%`;
  document.getElementById("vib-pct-n").textContent = `${n.toFixed(0)}%`;
  document.getElementById("vib-pct-a").textContent = `${a.toFixed(0)}%`;
  alertForLabel(data.label);
  const led =
    data.label === "DANGER" ? "red" : data.label === "ABNORMAL" || data.label === "CAUTION" ? "yellow" : "green";
  updateLedUi(led);
}

function drawWave(series) {
  const canvas = document.getElementById("vib-canvas");
  if (!canvas || !series) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fafbfc";
  ctx.fillRect(0, 0, w, h);

  const lanes = [
    { key: "x", color: "#1f6feb", y0: h * 0.2 },
    { key: "y", color: "#1a7f4b", y0: h * 0.5 },
    { key: "z", color: "#c9372c", y0: h * 0.8 },
  ];

  lanes.forEach(({ key, color, y0 }) => {
    const arr = series[key] || [];
    if (!arr.length) return;
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const span = Math.max(max - min, 1e-6);
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * (w - 24) + 12;
      const y = y0 - ((v - min) / span - 0.5) * (h * 0.22);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "12px JetBrains Mono, monospace";
    ctx.fillText(key.toUpperCase(), 12, y0 - h * 0.12);
  });
}
