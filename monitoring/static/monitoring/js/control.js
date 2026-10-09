import { alertForLabel, stopAlerts } from "./sounds.js?v=control1";
import { renderJoinQr } from "./qr-scan.js?v=control1";

const state = {
  mode: "upload",
  liveActive: false,
};

let file = null;
let livePoll = null;
let pollBusy = false;

export function initControlCenter() {
  const drop = document.getElementById("cc-drop");
  const fileInput = document.getElementById("cc-file");
  const player = document.getElementById("cc-player");
  const analyzeBtn = document.getElementById("cc-analyze");
  const status = document.getElementById("cc-upload-status");
  const machineBtn = document.getElementById("cc-machine");

  function paintMachine(on) {
    if (!machineBtn) return;
    machineBtn.setAttribute("aria-pressed", on ? "true" : "false");
    machineBtn.textContent = on ? "Machine on" : "Machine off";
    machineBtn.classList.toggle("primary", !!on);
  }

  machineBtn?.addEventListener("click", async () => {
    const next = machineBtn.getAttribute("aria-pressed") !== "true";
    machineBtn.disabled = true;
    try {
      const res = await fetch("/api/hardware/mode/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ machine_on: next }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not set the machine");
      paintMachine(!!data.hardware?.machine_on);
    } catch (err) {
      showLiveMsg(String(err.message || err), false);
    } finally {
      machineBtn.disabled = false;
    }
  });

  fetch("/api/hardware/status/")
    .then((res) => res.json())
    .then((data) => paintMachine(!!data.hardware?.machine_on))
    .catch(() => {});

  function setFile(f) {
    if (!f) return;
    if (!f.type.startsWith("video/") && !/\.(mp4|mov|webm)$/i.test(f.name)) {
      status.textContent = "Please choose an mp4, mov, or webm video";
      return;
    }
    file = f;
    if (player.src) URL.revokeObjectURL(player.src);
    player.src = URL.createObjectURL(f);
    player.classList.add("visible");
    status.textContent = f.name;
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) setFile(fileInput.files[0]);
    fileInput.value = "";
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  });

  analyzeBtn.addEventListener("click", () => analyzeUpload(analyzeBtn, status));
  document.getElementById("cc-mode-upload").addEventListener("click", () => setMode("upload"));
  document.getElementById("cc-mode-live").addEventListener("click", () => setMode("live"));
  document.getElementById("cc-show-qr").addEventListener("click", () => showQrSession());
  document.getElementById("cc-stop").addEventListener("click", () => stopLive());
  if (new URLSearchParams(location.search).get("live") === "1") setMode("live", { attach: true });
}

function setMode(mode, { attach = false } = {}) {
  const leavingLive = mode !== "live" && state.liveActive;
  state.mode = mode;
  if (leavingLive) stopLive();
  document.getElementById("cc-mode-upload").classList.toggle("active", mode === "upload");
  document.getElementById("cc-mode-live").classList.toggle("active", mode === "live");
  document.getElementById("cc-upload-controls").hidden = mode === "live";
  document.getElementById("cc-live-controls").hidden = mode !== "live";
  document.getElementById("cc-live-preview").hidden = true;
  document.getElementById("cc-mode-hint").textContent =
    mode === "live"
      ? "Live SAMS starts only after the phone scans this QR."
      : "Drop a workplace clip, or go live after the phone scans the QR.";
  const pill = document.getElementById("cc-status-pill");
  if (mode === "upload") {
    if (pill) {
      pill.textContent = "upload";
      pill.classList.remove("live");
    }
    stopAlerts();
    document.getElementById("cc-qr-panel")?.classList.remove("linked");
  } else if (attach) {
    attachLive();
  } else {
    showQrSession();
  }
}

async function attachLive() {
  try {
    const res = await fetch("/api/hardware/camera/");
    const data = await res.json();
    if (!data.camera?.live) {
      showQrSession();
      return;
    }
  } catch {
    showQrSession();
    return;
  }
  state.liveActive = true;
  document.getElementById("cc-show-qr").disabled = true;
  document.getElementById("cc-stop").disabled = false;
  document.getElementById("cc-qr-panel")?.classList.add("linked");
  showLiveMsg("Phone linked — waiting for frames…");
  clearInterval(livePoll);
  livePoll = setInterval(pollLive, 1500);
  pollLive();
}

function showLiveMsg(msg, ok = true) {
  const el = document.getElementById("cc-live-msg");
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
  el.style.color = ok ? "#0b6e4f" : "#e03121";
}

async function analyzeUpload(analyzeBtn, status) {
  if (!file) {
    status.textContent = "Choose a video first";
    return;
  }
  status.textContent = "Running person, distance, and PPE…";
  analyzeBtn.disabled = true;
  const form = new FormData();
  form.append("video", file);
  form.append("max_frames", "8");
  try {
    const res = await fetch("/api/vision/video/", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      status.textContent = data.error || "Analyze failed";
      return;
    }
    renderFrames(data);
    const frames = data.frames || [];
    const chosen =
      [...frames].reverse().find((f) => f.prediction?.worker_detected) || frames[frames.length - 1];
    if (chosen?.prediction) {
      await fuseVision(chosen.prediction);
    }
    status.textContent = `${data.frame_count_sampled || frames.length} frames · SAMS decision updated`;
    if (data.recording_url) {
      const link = document.createElement("a");
      link.href = data.recording_url;
      link.textContent = "Open in archive";
      link.style.marginLeft = "0.6rem";
      status.append(" · ", link);
    }
  } catch (err) {
    status.textContent = String(err);
  } finally {
    analyzeBtn.disabled = false;
  }
}

async function fuseVision(prediction) {
  const vision = {
    label: prediction.label,
    confidence: prediction.confidence,
    worker_detected: prediction.worker_detected,
    person_count: prediction.person_count,
    worker_distance_est: prediction.worker_distance_est,
    distance_source: prediction.distance_source,
    people: prediction.people,
    ppe: prediction.ppe,
    ppe_summary: prediction.ppe_summary,
  };
  const res = await fetch("/api/control/fuse/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vision }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Fusion failed");
  renderDecision(data);
}

async function showQrSession() {
  const showBtn = document.getElementById("cc-show-qr");
  const stopBtn = document.getElementById("cc-stop");
  if (showBtn) showBtn.disabled = true;
  showLiveMsg("Creating QR session…");
  try {
    const res = await fetch("/api/hardware/camera/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "start" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    const joinUrl = data.join_url || data.camera?.join_url;
    if (!joinUrl) throw new Error("No join URL — is the server bound to 0.0.0.0?");
    await renderJoinQr(document.getElementById("cc-qr"), joinUrl);
    const urlEl = document.getElementById("cc-join-url");
    if (urlEl) {
      urlEl.hidden = false;
      urlEl.textContent = joinUrl;
    }
    state.liveActive = true;
    if (stopBtn) stopBtn.disabled = false;
    const pill = document.getElementById("cc-status-pill");
    if (pill) {
      pill.textContent = "waiting · scan QR";
      pill.classList.remove("live");
    }
    showLiveMsg("Waiting for the phone to scan the QR…");
    clearInterval(livePoll);
    livePoll = setInterval(pollLive, 1500);
    pollLive();
  } catch (err) {
    state.liveActive = false;
    if (showBtn) showBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    showLiveMsg(String(err.message || err), false);
  }
}

async function stopLive() {
  clearInterval(livePoll);
  livePoll = null;
  state.liveActive = false;
  const showBtn = document.getElementById("cc-show-qr");
  const stopBtn = document.getElementById("cc-stop");
  if (showBtn) showBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
  stopAlerts();
  try {
    await fetch("/api/hardware/camera/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop" }),
    });
  } catch {
    /* ignore */
  }
  document.getElementById("cc-live-preview").hidden = true;
  showLiveMsg("Disconnected — scan a new QR to go live again.");
}

async function pollLive() {
  if (!state.liveActive || pollBusy) return;
  pollBusy = true;
  try {
    const res = await fetch("/api/control/live/");
    const data = await res.json();
    const scanned = !!(data.camera?.phone_joined || data.camera?.has_frame);
    const pill = document.getElementById("cc-status-pill");
    if (!scanned) {
      if (pill) pill.textContent = "waiting · scan QR";
      showLiveMsg("Waiting for the phone to scan the QR…");
      return;
    }
    if (pill) {
      pill.textContent = "live · phone";
      pill.classList.add("live");
    }
    document.getElementById("cc-qr-panel")?.classList.add("linked");
    const img = document.getElementById("cc-preview-img");
    const thumb = data.vision?.thumbnail;
    if (thumb && img) {
      img.src = thumb;
      document.getElementById("cc-live-preview").hidden = false;
      document.getElementById("cc-results").hidden = true;
    }
    if (data.ready === false || !data.fusion) {
      showLiveMsg(data.error || "Phone linked — waiting for frames…");
      return;
    }
    renderDecision(data);
    showLiveMsg(`Live · ${data.fusion.label}${data.fusion.reason ? " · " + data.fusion.reason : ""}`);
  } catch (err) {
    showLiveMsg(String(err.message || err), false);
  } finally {
    pollBusy = false;
  }
}

function renderDecision(data) {
  const fusion = data.fusion || {};
  const label = fusion.label || "—";
  const conf = fusion.confidence != null ? `${Math.round(fusion.confidence * 100)}%` : "—";
  document.getElementById("cc-fusion-label").textContent = label;
  document.getElementById("cc-fusion-conf").textContent = `Confidence ${conf}`;
  document.getElementById("cc-fusion-inline").textContent = label;
  const reason = document.getElementById("cc-reason");
  if (reason) reason.textContent = fusion.reason || "";
  const badge = document.getElementById("cc-fusion-badge");
  const hud = document.getElementById("cc-fusion-hud");
  for (const el of [badge, hud]) {
    if (!el) continue;
    el.classList.remove("state-NORMAL", "state-CAUTION", "state-DANGER", "state-DEGRADED");
    if (fusion.label) el.classList.add(`state-${fusion.label}`);
  }

  const vision = data.vision || {};
  document.getElementById("cc-vision").textContent = vision.label || "—";
  document.getElementById("cc-distance").textContent = formatDistance(
    vision.worker_detected ? vision.worker_distance_est : null,
    vision.distance_source,
  );
  document.getElementById("cc-zone").textContent = data.zone || "—";

  const vib = data.vibration;
  const vibEl = document.getElementById("cc-vibration");
  if (!vib || vib.ready === false || !vib.label) {
    vibEl.textContent = "No sensor";
  } else {
    vibEl.textContent = vib.label;
    vibEl.style.color = vib.abnormal ? "#e03121" : "#128a4b";
  }

  fillPpe(document.getElementById("cc-ppe"), vision.ppe);
  renderOthers("cc-people", vision.people);
  const probs = fusion.probabilities || {};
  for (const key of ["normal", "caution", "danger"]) {
    const name = key.toUpperCase();
    const pct = (probs[name] || 0) * 100;
    const bar = document.getElementById(`cc-bar-${key}`);
    const em = document.getElementById(`cc-pct-${key}`);
    if (bar) bar.style.width = `${pct}%`;
    if (em) em.textContent = fusion.label ? `${pct.toFixed(0)}%` : "—";
  }
  if (fusion.label) alertForLabel(fusion.label);
}

function formatDistance(metres, source) {
  if (metres == null || Number.isNaN(Number(metres))) return "—";
  const value = `${Number(metres).toFixed(1)} m`;
  if (source === "tof") return `${value} · sensor`;
  if (source === "vision") return `${value} · body`;
  return "—";
}

function renderOthers(id, people) {
  const el = document.getElementById(id);
  if (!el) return;
  const others = (people || []).filter((person) => !person.selected);
  if (!others.length) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = others
    .map((person) => {
      const dist =
        person.distance_m == null || (person.distance_source !== "tof" && person.distance_source !== "vision")
          ? ""
          : ` · ${Number(person.distance_m).toFixed(1)} m`;
      return `Person ${person.id}: ${person.summary || "—"}${dist}`;
    })
    .join("   ");
}

function fillPpe(root, ppe) {
  if (!root) return;
  root.querySelectorAll("[data-ppe]").forEach((row) => {
    const item = ppe?.items?.[row.dataset.ppe];
    const em = row.querySelector("em");
    if (em) em.textContent = item?.status || "—";
    row.classList.toggle("worn", item?.worn === true);
    row.classList.toggle("missing", item?.worn === false);
  });
}

function renderFrames(data) {
  const root = document.getElementById("cc-results");
  const frames = data.frames || [];
  root.hidden = false;
  if (!frames.length) {
    root.className = "frame-grid empty-state";
    root.textContent = "No frames extracted.";
    return;
  }
  root.className = "frame-grid";
  root.innerHTML = "";
  frames.forEach((f) => {
    const card = document.createElement("article");
    const pred = f.prediction || {};
    card.className = `frame-card${pred.worker_detected ? " unsafe" : ""}`;
    const dist =
      pred.worker_distance_est != null ? ` · ${Number(pred.worker_distance_est).toFixed(1)} m` : "";
    const ppe = pred.ppe?.summary && pred.ppe.summary !== "—" ? pred.ppe.summary : "";
    card.innerHTML = `
      <img src="${f.thumbnail}" alt="Frame ${f.index}" />
      <div class="meta">
        <strong>${pred.label || "Frame"}</strong>
        <span>${dist.replace(" · ", "") || "—"}${ppe ? ` · ${ppe}` : ""}</span>
      </div>
    `;
    root.appendChild(card);
  });
}
