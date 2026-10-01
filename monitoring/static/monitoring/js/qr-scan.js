/** Draw a QR code for the phone to scan (join URL). Uses vendored qrcodejs. */

let qrLibReady = false;

function loadQrLib() {
  if (qrLibReady && window.QRCode) return Promise.resolve(window.QRCode);
  if (window.QRCode) {
    qrLibReady = true;
    return Promise.resolve(window.QRCode);
  }
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/static/monitoring/vendor/qrcode.min.js";
    s.onload = () => {
      if (!window.QRCode) {
        reject(new Error("QR library unavailable"));
        return;
      }
      qrLibReady = true;
      resolve(window.QRCode);
    };
    s.onerror = () => reject(new Error("Could not load QR library"));
    document.head.appendChild(s);
  });
}

/**
 * @param {HTMLElement} mount - canvas or div container
 * @param {string} text - join URL
 */
export async function renderJoinQr(mount, text) {
  const QRCode = await loadQrLib();
  const host =
    mount.tagName === "CANVAS"
      ? (() => {
          const wrap = document.createElement("div");
          wrap.id = mount.id || "cam-qr";
          wrap.className = mount.className || "cam-qr";
          mount.replaceWith(wrap);
          return wrap;
        })()
      : mount;

  host.innerHTML = "";
  host.style.width = "280px";
  host.style.height = "280px";
  host.style.maxWidth = "100%";
  host.style.background = "#fff";
  host.style.borderRadius = "8px";
  host.style.display = "grid";
  host.style.placeItems = "center";

  // qrcodejs draws into the container (canvas or table)
  // eslint-disable-next-line no-new
  new QRCode(host, {
    text: String(text),
    width: 260,
    height: 260,
    colorDark: "#111111",
    colorLight: "#ffffff",
    correctLevel: QRCode.CorrectLevel.M,
  });
}
