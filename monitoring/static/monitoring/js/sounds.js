/** Continuous industrial alert tones via Web Audio. */

let ctx = null;
let enabled = true;
let currentLabel = "NORMAL";
let loopTimer = null;
let masterGain = null;

export function setSoundEnabled(on) {
  enabled = !!on;
  if (!enabled) stopLoop();
  else if (currentLabel === "CAUTION" || currentLabel === "DANGER" || currentLabel === "ABNORMAL") {
    startLoop(currentLabel);
  }
}

function ac() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  if (!masterGain) {
    masterGain = ctx.createGain();
    masterGain.gain.value = 0.14;
    masterGain.connect(ctx.destination);
  }
  return ctx;
}

function stopLoop() {
  if (loopTimer) {
    clearInterval(loopTimer);
    loopTimer = null;
  }
}

function blip({ freq, duration, type = "square", gain = 0.9, slideTo = null }) {
  if (!enabled || !masterGain) return;
  const audio = ac();
  const osc = audio.createOscillator();
  const g = audio.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, audio.currentTime);
  if (slideTo != null) {
    osc.frequency.linearRampToValueAtTime(slideTo, audio.currentTime + duration);
  }
  g.gain.setValueAtTime(0.0001, audio.currentTime);
  g.gain.exponentialRampToValueAtTime(gain, audio.currentTime + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + duration);
  osc.connect(g);
  g.connect(masterGain);
  osc.start();
  osc.stop(audio.currentTime + duration + 0.02);
}

function cautionPulse() {
  blip({ freq: 880, duration: 0.16, type: "square", gain: 0.75 });
  setTimeout(() => blip({ freq: 660, duration: 0.18, type: "square", gain: 0.7 }), 170);
}

function dangerPulse() {
  blip({ freq: 520, duration: 0.22, type: "sawtooth", gain: 0.95, slideTo: 280 });
  setTimeout(() => blip({ freq: 280, duration: 0.18, type: "square", gain: 0.85, slideTo: 460 }), 240);
  setTimeout(() => blip({ freq: 700, duration: 0.12, type: "square", gain: 0.8 }), 480);
}

function startLoop(label) {
  stopLoop();
  ac();
  const mode = label === "DANGER" ? "DANGER" : "CAUTION";
  const tick = () => {
    if (!enabled) return;
    if (mode === "DANGER") dangerPulse();
    else cautionPulse();
  };
  tick();
  loopTimer = setInterval(tick, mode === "DANGER" ? 700 : 900);
}

export function alertForLabel(label) {
  if (!label) return;
  const prev = currentLabel;
  currentLabel = label;

  if (!enabled || label === "NORMAL") {
    stopLoop();
    return;
  }

  if (label === "CAUTION" || label === "ABNORMAL" || label === "DANGER") {
    if (prev !== label || !loopTimer) startLoop(label);
  } else {
    stopLoop();
  }
}

export function stopAlerts() {
  currentLabel = "NORMAL";
  stopLoop();
}
