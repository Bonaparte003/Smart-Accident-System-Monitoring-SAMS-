import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { alertForLabel } from "./sounds.js?v=phoneqr3";

const MACHINE_POS = new THREE.Vector3(0, 0.55, 0);

const state = {
  distance: 1.8,
  approaching: true,
  machineActive: true,
  abnormalVib: false,
  wearings: {
    hard_hat: true,
    vest: true,
    goggles: true,
    gloves: true,
  },
  overlays: { vision: true, vibration: true, fusion: true },
  zones: { caution: 2.0, danger: 0.8 },
  riskLabel: "NORMAL",
  animating: false,
  anim: null,
  ready: false,
};

const els = {};
let scene, camera, renderer, controls;
let cctvCamera, previewRenderer;
let workerGroup, wearMeshes, machineMesh, machineLight;
let cautionZone, dangerZone, cameraPole, visionRay, vibWave;
let distanceLine, distanceSprite;
let pendingTimer = null;
let lastFetch = 0;

function $(id) {
  return document.getElementById(id);
}

function makeLabelCanvas(text, color = "#1f6feb") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillRect(8, 10, 240, 44);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(8, 10, 240, 44);
  ctx.fillStyle = color;
  ctx.font = "600 22px DM Sans, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  return canvas;
}

function makeTextSprite(text, color) {
  const canvas = makeLabelCanvas(text, color);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(1.4, 0.35, 1);
  sprite.userData.canvas = canvas;
  sprite.userData.texture = tex;
  return sprite;
}

function updateSpriteText(sprite, text, color) {
  const canvas = sprite.userData.canvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.fillRect(8, 10, 240, 44);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(8, 10, 240, 44);
  ctx.fillStyle = color;
  ctx.font = "600 22px DM Sans, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 32);
  sprite.userData.texture.needsUpdate = true;
}

function createWorker() {
  const g = new THREE.Group();
  // Base body (coveralls)
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2f343a, roughness: 0.7, metalness: 0 });
  const skinMat = new THREE.MeshStandardMaterial({ color: 0xc4a484, roughness: 0.65, metalness: 0 });
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), skinMat);
  head.position.y = 1.55;
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.14, 0.45, 6, 12), bodyMat);
  torso.position.y = 1.05;
  const legL = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.45, 4, 8), bodyMat);
  legL.position.set(-0.08, 0.35, 0);
  const legR = legL.clone();
  legR.position.x = 0.08;
  const armL = new THREE.Mesh(new THREE.CapsuleGeometry(0.045, 0.35, 4, 8), bodyMat);
  armL.position.set(-0.24, 1.15, 0);
  armL.rotation.z = 0.25;
  const armR = armL.clone();
  armR.position.x = 0.24;
  armR.rotation.z = -0.25;
  g.add(head, torso, legL, legR, armL, armR);

  // —— PPE wearings ——
  const hatMat = new THREE.MeshStandardMaterial({ color: 0xe6a100, roughness: 0.45, metalness: 0.05 });
  const hardHat = new THREE.Group();
  const dome = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2), hatMat);
  dome.position.y = 1.62;
  const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.03, 20), hatMat);
  brim.position.y = 1.61;
  const ridge = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.18), hatMat);
  ridge.position.y = 1.72;
  hardHat.add(dome, brim, ridge);

  const vestMat = new THREE.MeshStandardMaterial({ color: 0xe6a100, roughness: 0.55, metalness: 0 });
  const stripeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.1 });
  const vest = new THREE.Group();
  const vestBody = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.15, 0.42, 16, 1, true), vestMat);
  vestBody.position.y = 1.08;
  const stripe1 = new THREE.Mesh(new THREE.TorusGeometry(0.155, 0.015, 8, 24), stripeMat);
  stripe1.position.y = 1.18;
  stripe1.rotation.x = Math.PI / 2;
  const stripe2 = stripe1.clone();
  stripe2.position.y = 0.98;
  vest.add(vestBody, stripe1, stripe2);

  const goggleMat = new THREE.MeshStandardMaterial({
    color: 0x111111,
    roughness: 0.3,
    metalness: 0.2,
    transparent: true,
    opacity: 0.85,
  });
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x88ccee,
    roughness: 0.1,
    metalness: 0.4,
    transparent: true,
    opacity: 0.55,
  });
  const goggles = new THREE.Group();
  const band = new THREE.Mesh(new THREE.TorusGeometry(0.125, 0.012, 8, 24), goggleMat);
  band.position.y = 1.56;
  band.rotation.x = Math.PI / 2;
  const lensL = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 12), lensMat);
  lensL.position.set(-0.05, 1.56, 0.1);
  lensL.scale.set(1, 0.7, 0.45);
  const lensR = lensL.clone();
  lensR.position.x = 0.05;
  goggles.add(band, lensL, lensR);

  const gloveMat = new THREE.MeshStandardMaterial({ color: 0xe6a100, roughness: 0.55, metalness: 0 });
  const gloves = new THREE.Group();
  const gloveL = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 12), gloveMat);
  gloveL.position.set(-0.28, 0.92, 0.02);
  gloveL.scale.set(1, 1.15, 0.85);
  const gloveR = gloveL.clone();
  gloveR.position.x = 0.28;
  gloves.add(gloveL, gloveR);

  g.add(hardHat, vest, goggles, gloves);
  const label = makeTextSprite("Worker 01", "#111111");
  label.position.set(0, 1.95, 0);
  g.add(label);

  g.userData.wearings = {
    hard_hat: hardHat,
    vest,
    goggles,
    gloves,
  };
  return g;
}

function createMachine() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 1.1, 0.9),
    new THREE.MeshStandardMaterial({ color: 0x3a3f48, roughness: 0.5, metalness: 0.25 })
  );
  body.position.y = 0.55;
  body.castShadow = true;
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.15, 0.7),
    new THREE.MeshStandardMaterial({ color: 0x555c66, roughness: 0.4, metalness: 0.3 })
  );
  top.position.y = 1.18;
  const light = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.55 })
  );
  light.position.set(0.4, 1.18, 0.4);
  g.add(body, top, light);
  g.userData.light = light;
  return g;
}

function createZone(radius, color, opacity) {
  const geo = new THREE.CircleGeometry(radius, 48);
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.02;
  const edge = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: 64 }, (_, i) => {
        const a = (i / 64) * Math.PI * 2;
        return new THREE.Vector3(Math.cos(a) * radius, 0.03, Math.sin(a) * radius);
      })
    ),
    new THREE.LineBasicMaterial({ color, transparent: false })
  );
  const group = new THREE.Group();
  group.add(mesh, edge);
  return group;
}

function createCameraPole() {
  const g = new THREE.Group();
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.03, 0.04, 2.2, 8),
    new THREE.MeshStandardMaterial({ color: 0x8b93a0 })
  );
  pole.position.y = 1.1;
  const cam = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xe6a100, emissive: 0xe6a100, emissiveIntensity: 0.45 })
  );
  cam.position.y = 2.2;
  g.add(pole, cam);
  g.position.set(-2.4, 0, 1.6);
  g.userData.camPos = new THREE.Vector3(-2.4, 2.2, 1.6);
  return g;
}

function dashedLine(color) {
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const mat = new THREE.LineDashedMaterial({
    color,
    dashSize: 0.14,
    gapSize: 0.09,
    transparent: false,
  });
  const line = new THREE.Line(geo, mat);
  line.computeLineDistances();
  return line;
}

function createVibWave() {
  const points = Array.from({ length: 40 }, (_, i) => new THREE.Vector3(i * 0.04, 0, 0));
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: 0xd946ef, transparent: false });
  const line = new THREE.Line(geo, mat);
  line.position.set(0.7, 0.9, 0);
  return line;
}

function initScene() {
  els.viewport = $("viewport");
  if (!els.viewport || state.ready) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f6f8);
  scene.fog = new THREE.Fog(0xf4f6f8, 16, 36);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(5.5, 4.2, 6.5);

  cctvCamera = new THREE.PerspectiveCamera(50, 16 / 10, 0.1, 50);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  els.viewport.appendChild(renderer.domElement);

  const previewCanvas = $("cam-preview-canvas");
  if (previewCanvas) {
    previewRenderer = new THREE.WebGLRenderer({
      canvas: previewCanvas,
      antialias: true,
      alpha: false,
    });
    previewRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    previewRenderer.setSize(previewCanvas.clientWidth || 320, previewCanvas.clientHeight || 200, false);
    previewRenderer.setClearColor(0x1a1a1a, 1);
  }

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.8, 1.2);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xdde2ea, 0.95));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(4, 8, 3);
  dir.castShadow = true;
  scene.add(dir);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const grid = new THREE.GridHelper(20, 40, 0xd8dde6, 0xe8ebf0);
  grid.position.y = 0.01;
  scene.add(grid);

  machineMesh = createMachine();
  machineLight = machineMesh.userData.light;
  scene.add(machineMesh);

  cautionZone = createZone(state.zones.caution, 0xe6a100, 0.45);
  dangerZone = createZone(state.zones.danger, 0xe03121, 0.55);
  scene.add(cautionZone, dangerZone);

  workerGroup = createWorker();
  wearMeshes = workerGroup.userData.wearings;
  scene.add(workerGroup);

  cameraPole = createCameraPole();
  scene.add(cameraPole);

  visionRay = dashedLine(0xe6a100);
  distanceLine = dashedLine(0xe03121);
  scene.add(visionRay, distanceLine);

  distanceSprite = makeTextSprite("1.8 m", "#e03121");
  scene.add(distanceSprite);

  vibWave = createVibWave();
  scene.add(vibWave);

  window.addEventListener("resize", onResize);
  onResize();
  animate();
  state.ready = true;
}

function syncWorkerTransform() {
  workerGroup.position.set(0, 0, state.distance);
  workerGroup.rotation.y = state.approaching ? Math.PI : Math.PI * 0.5;
  if (wearMeshes) {
    Object.entries(state.wearings).forEach(([key, on]) => {
      if (wearMeshes[key]) wearMeshes[key].visible = !!on;
    });
  }
}

function syncCctvCamera() {
  if (!cctvCamera || !cameraPole || !workerGroup) return;
  const camPos = cameraPole.userData.camPos.clone();
  cctvCamera.position.copy(camPos);
  const lookAt = workerGroup.position.clone().add(new THREE.Vector3(0, 1.35, 0));
  cctvCamera.lookAt(lookAt);
}

function updateOverlays(t) {
  const workerHead = workerGroup.position.clone().add(new THREE.Vector3(0, 1.4, 0));
  const machineFace = MACHINE_POS.clone().add(new THREE.Vector3(0, 0.4, 0.5));
  const camPos = cameraPole.userData.camPos;

  visionRay.visible = state.overlays.vision;
  if (visionRay.visible) {
    visionRay.geometry.setFromPoints([camPos, workerHead]);
    visionRay.computeLineDistances();
  }

  const showFusion = state.overlays.fusion;
  distanceLine.visible = showFusion;
  distanceSprite.visible = showFusion;
  if (showFusion) {
    distanceLine.geometry.setFromPoints([
      workerGroup.position.clone().add(new THREE.Vector3(0, 0.08, 0)),
      machineFace,
    ]);
    distanceLine.computeLineDistances();
    const mid = workerGroup.position.clone().lerp(machineFace, 0.5);
    mid.y = 0.55;
    distanceSprite.position.copy(mid);
    updateSpriteText(distanceSprite, `${state.distance.toFixed(1)} m`, "#e03121");
  }

  vibWave.visible = state.overlays.vibration && state.machineActive;
  if (vibWave.visible) {
    const amp = state.abnormalVib ? 0.22 : 0.06;
    const freq = state.abnormalVib ? 14 : 5;
    const pos = vibWave.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      pos.setXYZ(i, i * 0.04, Math.sin(t * freq + i * 0.45) * amp, 0);
    }
    pos.needsUpdate = true;
  }

  machineLight.material.emissiveIntensity = state.machineActive ? 0.9 + Math.sin(t * 4) * 0.2 : 0.1;
  if (!state.machineActive) {
    machineLight.material.color.setHex(0x555555);
    machineLight.material.emissive.setHex(0x222222);
  } else if (state.riskLabel === "DANGER") {
    machineLight.material.color.setHex(0xe03121);
    machineLight.material.emissive.setHex(0xe03121);
    machineLight.material.emissiveIntensity = 1.1 + Math.sin(t * 10) * 0.35;
  } else if (state.riskLabel === "CAUTION") {
    machineLight.material.color.setHex(0xe6a100);
    machineLight.material.emissive.setHex(0xe6a100);
    machineLight.material.emissiveIntensity = 0.95 + Math.sin(t * 5) * 0.3;
  } else {
    machineLight.material.color.setHex(0xffffff);
    machineLight.material.emissive.setHex(0xffffff);
  }

  if (cautionZone) {
    const pulse = state.riskLabel === "CAUTION" ? 0.55 + Math.sin(t * 3) * 0.15 : 0.45;
    cautionZone.children[0].material.opacity = pulse;
  }
}

function onResize() {
  if (!els.viewport || !renderer) return;
  const w = els.viewport.clientWidth;
  const h = els.viewport.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);

  const previewCanvas = $("cam-preview-canvas");
  if (previewRenderer && previewCanvas) {
    const pw = previewCanvas.clientWidth || 320;
    const ph = previewCanvas.clientHeight || 200;
    previewRenderer.setSize(pw, ph, false);
    cctvCamera.aspect = pw / Math.max(ph, 1);
    cctvCamera.updateProjectionMatrix();
  }
}

function animate(now = 0) {
  requestAnimationFrame(animate);
  if (!state.ready) return;
  const t = now * 0.001;

  if (state.animating && state.anim) {
    const { from, to, start, duration, approaching } = state.anim;
    const u = Math.min(1, (performance.now() - start) / duration);
    const eased = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    state.distance = from + (to - from) * eased;
    state.approaching = approaching;
    els.distance.value = state.distance;
    els.distanceValue.textContent = `${state.distance.toFixed(1)} m`;
    els.approaching.checked = approaching;
    if (u >= 1) {
      state.animating = false;
      state.anim = null;
    }
    schedulePredict();
  }

  syncWorkerTransform();
  syncCctvCamera();
  updateOverlays(t);
  controls.update();
  renderer.render(scene, camera);
  if (previewRenderer && cctvCamera) {
    previewRenderer.render(scene, cctvCamera);
  }
}

function setCameraPreset(name) {
  controls.enabled = name === "orbit";
  const presets = {
    orbit: { pos: [5.5, 4.2, 6.5], target: [0, 0.8, 1.2] },
    top: { pos: [0.01, 9, 0.01], target: [0, 0, 1] },
    side: { pos: [7, 2.2, 1.5], target: [0, 0.8, 1] },
    operator: { pos: [-1.2, 1.6, 4.5], target: [0, 0.9, 0.5] },
    cctv: {
      pos: cameraPole?.userData.camPos?.toArray?.() || [-2.4, 2.2, 1.6],
      target: [0, 1.2, state.distance],
    },
  };
  const p = presets[name];
  if (!p) return;
  camera.position.set(...p.pos);
  controls.target.set(...p.target);
  controls.update();
}

function schedulePredict() {
  clearTimeout(pendingTimer);
  pendingTimer = setTimeout(predict, 80);
}

async function predict() {
  const now = performance.now();
  if (now - lastFetch < 50) {
    schedulePredict();
    return;
  }
  lastFetch = now;
  try {
    const res = await fetch("/api/simulate/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        worker_distance: state.distance,
        approaching_machine: state.approaching,
        machine_active: state.machineActive,
        abnormal_vibration: state.abnormalVib,
        hard_hat: state.wearings.hard_hat,
        vest: state.wearings.vest,
        goggles: state.wearings.goggles,
        gloves: state.wearings.gloves,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    applyHud(await res.json());
  } catch (err) {
    console.error(err);
  }
}

function applyHud(data) {
  const hud = data.hud || {};
  const fusion = hud.fusion || {};
  const label = fusion.status || "—";

  if (data.hardware?.mode === "live" || data.live_vibration) {
    syncLiveVibrationUi(data);
  }

  els.hudVision.textContent = hud.vision?.status || "—";
  if (els.hudPpe) {
    const summary = hud.ppe?.summary || "—";
    els.hudPpe.textContent = summary;
    els.hudPpe.style.color = hud.ppe?.complete ? "#128a4b" : summary === "—" ? "" : "#e03121";
  }
  if (els.ppeTrack && hud.ppe?.items) {
    Object.entries(hud.ppe.items).forEach(([key, item]) => {
      const row = els.ppeTrack.querySelector(`[data-ppe="${key}"]`);
      if (!row) return;
      const em = row.querySelector("em");
      if (em) em.textContent = item.status;
      row.classList.toggle("worn", item.status === "Worn");
      row.classList.toggle("missing", item.status === "Missing");
    });
  }
  if (els.camMeta) {
    const missing = hud.ppe?.missing?.length || 0;
    els.camMeta.textContent = hud.ppe?.complete
      ? "PPE: complete"
      : missing
        ? `PPE missing: ${missing}`
        : "PPE: —";
  }
  els.hudVibration.textContent = hud.vibration?.status || "—";
  els.hudZone.textContent = hud.zone || "—";
  els.hudFusion.textContent = label;
  els.fusionLabel.textContent = label;
  const confPct = fusion.confidence != null ? Math.round(fusion.confidence * 100) : null;
  els.fusionConf.textContent = confPct != null ? `Confidence ${confPct}%` : "Confidence —";
  els.fusionBadge.className = `fusion-chip state-${label}`;
  els.hudFusionTile.className = `hud hud-fusion state-${label}`;
  state.riskLabel = label;

  if (els.cautionBanner) {
    const showCaution = label === "CAUTION";
    els.cautionBanner.hidden = !showCaution;
    if (showCaution && els.cautionBannerText) {
      const zone = hud.zone || "Caution";
      const ppe = hud.ppe?.complete === false ? " PPE incomplete." : "";
      els.cautionBannerText.textContent =
        zone === "Caution" || zone === "Danger"
          ? `Worker in ${zone.toLowerCase()} zone — proceed carefully.${ppe}`
          : `Elevated risk detected — proceed carefully.${ppe}`;
    }
  }

  const p = fusion.probabilities || {};
  const n = (p.NORMAL || 0) * 100;
  const c = (p.CAUTION || 0) * 100;
  const d = (p.DANGER || 0) * 100;
  els.barNormal.style.width = `${n}%`;
  els.barCaution.style.width = `${c}%`;
  els.barDanger.style.width = `${d}%`;
  els.pctNormal.textContent = `${n.toFixed(0)}%`;
  els.pctCaution.textContent = `${c.toFixed(0)}%`;
  els.pctDanger.textContent = `${d.toFixed(0)}%`;

  alertForLabel(label);
}

function runScenario(id) {
  const scripts = {
    normal_pass: { from: 4.5, to: 2.4, duration: 2800, approaching: false, vib: false, machine: true },
    cautious_approach: {
      from: 3.8,
      to: 1.55,
      duration: 5200,
      approaching: true,
      vib: false,
      machine: true,
      wearings: { hard_hat: true, vest: true, goggles: true, gloves: true },
    },
    slow_approach: { from: 3.5, to: 1.5, duration: 4200, approaching: true, vib: false, machine: true },
    fast_approach: { from: 3.2, to: 1.0, duration: 1600, approaching: true, vib: false, machine: true },
    danger_entry: { from: 2.2, to: 0.5, duration: 2400, approaching: true, vib: true, machine: true },
  };
  const s = scripts[id];
  if (!s) return;
  state.abnormalVib = s.vib;
  state.machineActive = s.machine;
  els.abnormalVib.checked = s.vib;
  els.machineActive.checked = s.machine;
  if (s.wearings) {
    Object.entries(s.wearings).forEach(([key, on]) => {
      state.wearings[key] = on;
    });
    if (els.wearHardHat) els.wearHardHat.checked = state.wearings.hard_hat;
    if (els.wearVest) els.wearVest.checked = state.wearings.vest;
    if (els.wearGoggles) els.wearGoggles.checked = state.wearings.goggles;
    if (els.wearGloves) els.wearGloves.checked = state.wearings.gloves;
  }
  state.animating = true;
  state.anim = {
    from: s.from,
    to: s.to,
    start: performance.now(),
    duration: s.duration,
    approaching: s.approaching,
  };
}

function cacheEls() {
  els.distance = $("distance");
  els.distanceValue = $("distance-value");
  els.approaching = $("approaching");
  els.machineActive = $("machine-active");
  els.abnormalVib = $("abnormal-vib");
  els.wearHardHat = $("wear-hard-hat");
  els.wearVest = $("wear-vest");
  els.wearGoggles = $("wear-goggles");
  els.wearGloves = $("wear-gloves");
  els.fusionBadge = $("fusion-badge");
  els.fusionLabel = $("fusion-label");
  els.fusionConf = $("fusion-conf");
  els.cautionBanner = $("caution-banner");
  els.cautionBannerText = $("caution-banner-text");
  els.hudVision = $("hud-vision-v");
  els.hudPpe = $("hud-ppe-v");
  els.hudVibration = $("hud-vibration-v");
  els.hudZone = $("hud-zone-v");
  els.hudFusion = $("hud-fusion-v");
  els.hudFusionTile = $("hud-fusion");
  els.ppeTrack = $("ppe-track");
  els.camMeta = $("cam-preview-meta");
  els.barNormal = $("bar-normal");
  els.barCaution = $("bar-caution");
  els.barDanger = $("bar-danger");
  els.pctNormal = $("pct-normal");
  els.pctCaution = $("pct-caution");
  els.pctDanger = $("pct-danger");
}

function bindUi() {
  els.distance.addEventListener("input", () => {
    state.distance = parseFloat(els.distance.value);
    els.distanceValue.textContent = `${state.distance.toFixed(1)} m`;
    schedulePredict();
  });
  els.approaching.addEventListener("change", () => {
    state.approaching = els.approaching.checked;
    schedulePredict();
  });
  els.machineActive.addEventListener("change", () => {
    state.machineActive = els.machineActive.checked;
    schedulePredict();
  });
  els.abnormalVib.addEventListener("change", () => {
    state.abnormalVib = els.abnormalVib.checked;
    schedulePredict();
  });

  const wearMap = [
    ["wearHardHat", "hard_hat"],
    ["wearVest", "vest"],
    ["wearGoggles", "goggles"],
    ["wearGloves", "gloves"],
  ];
  wearMap.forEach(([elKey, wearKey]) => {
    els[elKey]?.addEventListener("change", () => {
      state.wearings[wearKey] = els[elKey].checked;
      schedulePredict();
    });
  });

  document.querySelectorAll("#page-simulation [data-overlay]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.overlay;
      state.overlays[key] = !state.overlays[key];
      btn.classList.toggle("active", state.overlays[key]);
      if (key === "vision") {
        const preview = $("cam-preview");
        if (preview) preview.style.display = state.overlays.vision ? "" : "none";
      }
    });
  });
  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", () => setCameraPreset(btn.dataset.cam));
  });
  document.querySelectorAll("[data-scenario]").forEach((btn) => {
    btn.addEventListener("click", () => runScenario(btn.dataset.scenario));
  });

  window.addEventListener("sams:hardware-mode", (ev) => {
    applyHardwareMode(ev.detail || {});
  });
  fetch("/api/hardware/status/")
    .then((r) => r.json())
    .then((d) => applyHardwareMode(d.hardware || {}))
    .catch(() => {});
}

function applyHardwareMode(hw) {
  const live = hw.mode === "live";
  const note = $("abnormal-vib-live-note");
  if (els.abnormalVib) {
    els.abnormalVib.disabled = live;
    if (live && hw.last_label) {
      els.abnormalVib.checked = hw.last_label === "ABNORMAL";
      state.abnormalVib = els.abnormalVib.checked;
    }
  }
  if (note) note.hidden = !live;
  if (live) schedulePredict();
}

function syncLiveVibrationUi(data) {
  const live = data.live_vibration;
  const note = $("abnormal-vib-live-note");
  if (els.abnormalVib) {
    els.abnormalVib.disabled = true;
    if (live) {
      els.abnormalVib.checked = !!live.abnormal;
      state.abnormalVib = !!live.abnormal;
    }
  }
  if (note) note.hidden = false;
}

export async function initSimulation(config) {
  cacheEls();
  if (config?.zones_m) state.zones = config.zones_m;
  if (config?.defaults) {
    state.distance = config.defaults.worker_distance ?? state.distance;
    state.approaching = !!config.defaults.approaching_machine;
    state.machineActive = !!config.defaults.machine_active;
    state.abnormalVib = !!config.defaults.abnormal_vibration;
    els.distance.value = state.distance;
    els.distanceValue.textContent = `${state.distance.toFixed(1)} m`;
    els.approaching.checked = state.approaching;
    els.machineActive.checked = state.machineActive;
    els.abnormalVib.checked = state.abnormalVib;
  }
  initScene();
  bindUi();
  onResize();
  await predict();
}

export function resizeSimulation() {
  onResize();
}
