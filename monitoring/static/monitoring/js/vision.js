import { stopAlerts } from "./sounds.js?v=uploadlive1";
import { renderJoinQr } from "./qr-scan.js?v=uploadlive1";

const state = {
  mode: "upload",
  liveActive: false,
};

let file = null;
let livePoll = null;
let pollBusy = false;

export function initVisionLab() {
  const drop = document.getElementById("vision-drop");
  const fileInput = document.getElementById("vision-file");
  const player = document.getElementById("vision-player");
  const analyzeBtn = document.getElementById("vision-analyze");
  const status = document.getElementById("vision-status");
  const results = document.getElementById("vision-results");
  const lightbox = document.getElementById("vision-lightbox");
  const lightboxImg = document.getElementById("vision-lightbox-img");
  const lightboxCaption = document.getElementById("vision-lightbox-caption");
  const lightboxClose = document.getElementById("vision-lightbox-close");

  function openZoom(src, caption) {
    lightboxImg.src = src;
    lightboxCaption.textContent = caption || "";
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeZoom() {
    lightbox.hidden = true;
    lightboxImg.removeAttribute("src");
    document.body.style.overflow = "";
  }

  lightboxClose.addEventListener("click", closeZoom);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox || e.target === lightboxImg) closeZoom();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) closeZoom();
  });

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

  analyzeBtn.addEventListener("click", async () => {
    if (!file) {
      status.textContent = "Choose a video first";
      return;
    }
    status.textContent = "Detecting people…";
    analyzeBtn.disabled = true;
    const form = new FormData();
    form.append("video", file);
    form.append("max_frames", "8");
    try {
      const res = await fetch("/api/vision/video/", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        const tip = data.torch?.available === false
          ? " Install ultralytics (pip install ultralytics) for YOLOv8 person detection."
          : "";
        status.textContent = (data.error || "Analyze failed") + tip;
        results.className = "frame-grid empty-state";
        results.textContent = data.error || "Analyze failed";
        return;
      }
      renderFrames(data, results, openZoom);
      const hits = (data.frames || []).filter((f) => f.prediction?.worker_detected);
      const backend = data.torch?.available ? "YOLOv8n" : "model unavailable";
      status.textContent = `${data.frame_count_sampled} frames · ${hits.length} with person · ${backend}`;
    } catch (err) {
      console.error(err);
      status.textContent = String(err);
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  document.getElementById("vision-mode-upload").addEventListener("click", () => setMode("upload"));
  document.getElementById("vision-mode-live").addEventListener("click", () => setMode("live"));
  document.getElementById("cam-show-qr").addEventListener("click", () => showQrSession());
  document.getElementById("cam-stop").addEventListener("click", () => stopLive());
}

function setMode(mode) {
  const leavingLive = mode !== "live" && state.liveActive;
  state.mode = mode;
  if (leavingLive) stopLive();
  document.getElementById("vision-mode-upload").classList.toggle("active", mode === "upload");
  document.getElementById("vision-mode-live").classList.toggle("active", mode === "live");
  document.getElementById("vision-upload-controls").hidden = mode === "live";
  document.getElementById("vision-live-controls").hidden = mode !== "live";
  document.getElementById("vision-live-preview").hidden = true;
  document.getElementById("vision-results").hidden = false;
  document.getElementById("vision-results-title").textContent =
    mode === "live" ? "Live phone feed" : "Detections";
  document.getElementById("vision-mode-hint").textContent =
    mode === "live"
      ? "Live starts only after the phone scans this QR."
      : "Drop a workplace clip to sample frames through YOLOv8 person detection.";
  if (mode === "upload") {
    updatePill({ live: false });
    stopAlerts();
    document.getElementById("qr-join-panel")?.classList.remove("linked");
    const results = document.getElementById("vision-results");
    if (results && !results.querySelector(".frame-card")) {
      results.className = "frame-grid empty-state";
      results.textContent = "Upload a video to see person detections on sampled frames.";
    }
  } else {
    showScanGate();
    showQrSession();
  }
}

function showScanGate() {
  const results = document.getElementById("vision-results");
  const preview = document.getElementById("vision-live-preview");
  if (preview) preview.hidden = true;
  if (!results) return;
  results.hidden = false;
  results.className = "frame-grid empty-state";
  results.textContent = "Scan the QR with your phone. Live detection starts after the scan.";
}

function showLiveFeed() {
  const results = document.getElementById("vision-results");
  const preview = document.getElementById("vision-live-preview");
  if (results) results.hidden = true;
  if (preview) preview.hidden = false;
}

function showCamMsg(msg, ok = true) {
  const el = document.getElementById("cam-status-msg");
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
  el.style.color = ok ? "#0b6e4f" : "#e03121";
}

function updatePill(cam) {
  const pill = document.getElementById("cam-status-pill");
  if (!pill) return;
  if (state.mode !== "live") {
    pill.textContent = "upload";
    pill.classList.remove("live");
    return;
  }
  if (cam?.phone_joined || cam?.has_frame) {
    pill.textContent = "live · phone";
    pill.classList.add("live");
  } else if (cam?.live || state.liveActive) {
    pill.textContent = "waiting · scan QR";
    pill.classList.remove("live");
  } else {
    pill.textContent = "show QR";
    pill.classList.remove("live");
  }
}

function setSessionUi(active) {
  const showBtn = document.getElementById("cam-show-qr");
  const stopBtn = document.getElementById("cam-stop");
  if (showBtn) showBtn.disabled = !!active;
  if (stopBtn) stopBtn.disabled = !active;
}

async function showQrSession() {
  setSessionUi(false);
  showCamMsg("Creating QR session…");
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
    if (joinUrl.startsWith("http://") && !joinUrl.includes("127.0.0.1")) {
      showCamMsg(
        "Warning: join URL is HTTP. Phone browsers block the camera on plain HTTP — restart with ./hardware/tools/hw.sh run (HTTPS).",
        false,
      );
    }
    await renderJoinQr(document.getElementById("cam-qr"), joinUrl);
    const urlEl = document.getElementById("cam-join-url");
    if (urlEl) {
      urlEl.hidden = false;
      urlEl.textContent = joinUrl;
    }
    state.liveActive = true;
    setSessionUi(true);
    updatePill(data.camera);
    showCamMsg("Waiting for phone to scan… (accept cert warning on phone)");
    clearInterval(livePoll);
    livePoll = setInterval(pollLive, 1500);
    pollLive();
  } catch (err) {
    state.liveActive = false;
    setSessionUi(false);
    showCamMsg(String(err.message || err), false);
  }
}

async function stopLive() {
  clearInterval(livePoll);
  livePoll = null;
  state.liveActive = false;
  setSessionUi(false);
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
  updatePill({ live: false });
  document.getElementById("qr-join-panel")?.classList.remove("linked");
  if (state.mode === "live") showScanGate();
  showCamMsg("Disconnected — scan a new QR to go live again.");
}

async function pollLive() {
  if (!state.liveActive || pollBusy) return;
  pollBusy = true;
  try {
    const res = await fetch("/api/vision/live/");
    const data = await res.json();
    updatePill(data.camera);
    const scanned = !!(data.camera?.phone_joined || data.camera?.has_frame);
    if (!scanned) {
      showScanGate();
      showCamMsg("Waiting for the phone to scan the QR…");
      return;
    }
    if (data.ready === false || !data.label) {
      showLiveFeed();
      showCamMsg(data.error || "Phone linked — waiting for frames…");
      return;
    }
    document.getElementById("qr-join-panel")?.classList.add("linked");
    showLiveFeed();
    renderLive(data);
  } catch (err) {
    showCamMsg(String(err.message || err), false);
  } finally {
    pollBusy = false;
  }
}

function renderLive(data) {
  const img = document.getElementById("cam-preview-img");
  if (data.thumbnail) img.src = data.thumbnail;
  document.getElementById("cam-label").textContent = data.label || "—";
  document.getElementById("cam-conf").textContent =
    data.confidence != null ? `${Math.round(data.confidence * 100)}%` : "—";
  const countEl = document.getElementById("cam-count");
  if (countEl) {
    countEl.textContent = data.person_count != null ? String(data.person_count) : "—";
    countEl.style.color = data.worker_detected ? "#e03121" : "#0b6e4f";
  }
  const distEl = document.getElementById("cam-distance");
  if (distEl) {
    const metres =
      data.worker_detected && data.worker_distance_est != null
        ? Number(data.worker_distance_est).toFixed(1)
        : null;
    const via =
      data.distance_source === "tof"
        ? " · sensor"
        : data.distance_source === "vision"
          ? " · body"
          : "";
    distEl.textContent = metres == null || !via ? "—" : `${metres} m${via}`;
  }
  fillPpeTrack(document.getElementById("cam-ppe"), data.ppe);
  const others = document.getElementById("cam-people");
  if (others) {
    const rest = (data.people || []).filter((person) => !person.selected);
    others.hidden = rest.length === 0;
    others.textContent = rest
      .map((person) => `Person ${person.id}: ${person.summary || "—"}`)
      .join("   ");
  }
  const ppeNote = data.ppe?.summary && data.ppe.summary !== "—" ? ` · ${data.ppe.summary}` : "";
  showCamMsg(
    data.worker_detected
      ? `Live · ${data.label} (${Math.round((data.confidence || 0) * 100)}%)${ppeNote}`
      : "Live · no person in frame",
  );
}

function fillPpeTrack(root, ppe) {
  if (!root) return;
  root.querySelectorAll("[data-ppe]").forEach((row) => {
    const item = ppe?.items?.[row.dataset.ppe];
    const em = row.querySelector("em");
    if (em) em.textContent = item?.status || "—";
    row.classList.toggle("worn", item?.worn === true);
    row.classList.toggle("missing", item?.worn === false);
  });
}

function renderFrames(data, root, openZoom) {
  const frames = data.frames || [];
  if (!frames.length) {
    root.className = "frame-grid empty-state";
    root.textContent = "No frames extracted.";
    return;
  }
  root.className = "frame-grid";
  root.innerHTML = "";
  frames.forEach((f) => {
    const card = document.createElement("article");
    const pred = f.prediction;
    const hit = !!pred?.worker_detected;
    card.className = `frame-card${hit ? " unsafe" : ""}`;
    const conf = pred ? Math.round(pred.confidence * 100) : null;
    const label = pred?.label || f.error || "No prediction";
    const confText = conf != null ? `${conf}%` : "—";
    const count = pred?.person_count != null ? ` · ${pred.person_count}p` : "";
    const dist =
      pred?.worker_distance_est != null ? ` · ${Number(pred.worker_distance_est).toFixed(1)} m` : "";
    const ppe = pred?.ppe?.summary && pred.ppe.summary !== "—" ? pred.ppe.summary : "";
    card.innerHTML = `
      <img src="${f.thumbnail}" alt="Frame ${f.index}" title="Click to zoom" />
      <div class="meta">
        <strong>${label}</strong>
        <span>${confText}${count}${dist}</span>
        ${ppe ? `<span>${ppe}</span>` : ""}
      </div>
    `;
    const img = card.querySelector("img");
    img.addEventListener("click", () => {
      openZoom(f.thumbnail, `Frame ${f.index} · ${label} · ${confText}`);
    });
    root.appendChild(card);
  });
}
