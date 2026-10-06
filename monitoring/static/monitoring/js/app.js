import { initControlCenter } from "./control.js?v=track1";
import { initVisionLab } from "./vision.js?v=alert1";
import { initVibrationLab } from "./vibration.js?v=uploadlive1";
import { setSoundEnabled } from "./sounds.js?v=control1";

const pages = ["control", "vision", "vibration", "logs"];
const crumbs = {
  control: { section: "Operations", page: "Control Center" },
  vision: { section: "Operations", page: "Vision" },
  vibration: { section: "Operations", page: "Vibration" },
  logs: { section: "System", page: "Activity log" },
};

function showPage(id) {
  pages.forEach((p) => {
    document.getElementById(`page-${p}`)?.classList.toggle("active", p === id);
    document.querySelector(`.nav-item[data-page="${p}"]`)?.classList.toggle("active", p === id);
  });
  const crumb = crumbs[id];
  if (crumb) {
    const eyebrow = document.getElementById("page-eyebrow");
    const pageCrumb = document.getElementById("page-crumb");
    if (eyebrow) eyebrow.textContent = crumb.section;
    if (pageCrumb) pageCrumb.textContent = crumb.page;
  }
  if (id === "logs") refreshLogs();
  history.replaceState(null, "", `#${id}`);
}

async function loadConfig() {
  const res = await fetch("/api/config/");
  const data = await res.json();
  if (data.vision_model) {
    const tag = document.getElementById("vision-model-tag");
    if (tag) {
      tag.textContent = data.vision_model.available
        ? data.vision_model.model_id
        : `${data.vision_model.model_id} · ultralytics missing`;
    }
  }
  if (data.vibration_model?.experiment_id) {
    const vibTag = document.getElementById("vib-model-tag");
    if (vibTag) vibTag.textContent = data.vibration_model.experiment_id;
  }
  return data;
}

function formatLogTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso || "");
  const pad = (value) => String(value).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

async function refreshLogs() {
  const root = document.getElementById("log-list");
  try {
    const res = await fetch("/api/logs/?limit=80");
    const data = await res.json();
    const events = data.events || [];
    if (!events.length) {
      root.className = "log-list empty-state";
      root.textContent = "No events yet. Use Control Center to generate activity.";
      return;
    }
    root.className = "log-list";
    root.innerHTML = events
      .map((e) => {
        const ts = formatLogTime(e.ts);
        return `<div class="log-item level-${e.level}">
          <span class="ts">${ts}</span>
          <span class="src">${e.source}</span>
          <span>${e.message}</span>
        </div>`;
      })
      .join("");
  } catch (err) {
    root.className = "log-list empty-state";
    root.textContent = String(err);
  }
}

function bindShell() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showPage(btn.dataset.page));
  });
  document.getElementById("sound-enabled").addEventListener("change", (e) => {
    setSoundEnabled(e.target.checked);
  });
  document.getElementById("logs-refresh").addEventListener("click", refreshLogs);
  document.getElementById("logs-clear").addEventListener("click", async () => {
    await fetch("/api/logs/clear/", { method: "POST" });
    refreshLogs();
  });

  const hash = location.hash.replace("#", "");
  if (pages.includes(hash)) showPage(hash);
}

async function main() {
  bindShell();
  initControlCenter();
  initVisionLab();
  initVibrationLab();
  try {
    await loadConfig();
  } catch (err) {
    console.error(err);
  }
  setInterval(() => {
    if (document.getElementById("page-logs")?.classList.contains("active")) {
      refreshLogs();
    }
  }, 2500);
}

main();
