// Camera preview, frame capture and face overlay drawing. Frames are captured into memory,
// sent to the local API and discarded; nothing here persists an image.
import { projectBox } from './model.js';

const SNAPSHOT_MAX_WIDTH = 640;
const SNAPSHOT_QUALITY = 0.82;

function cssToken(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export class CameraController {
  constructor(video, overlay) {
    this.video = video;
    this.overlay = overlay;
    this.stream = null;
    // Dimensions of the last JPEG that was sent to the API. Face boxes returned by the API
    // are expressed in this space, not in the camera's native resolution.
    this.frame = null;
  }

  async start() {
    if (this.stream) this.stop();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.stream = stream;
    this.video.srcObject = stream;
    try {
      await this.video.play();
    } catch (error) {
      // Never let a stream outlive the controller's bookkeeping.
      this.stop();
      throw error;
    }
    this.video.classList.add('live');
  }

  stop() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.frame = null;
    this.video.srcObject = null;
    this.video.classList.remove('live');
    this.clearOverlay();
  }

  /** Capture the current preview frame as a mirrored JPEG blob (matches the mirrored preview). */
  async snapshot() {
    if (!this.stream || this.video.readyState < 2) return null;
    const sourceW = this.video.videoWidth;
    const sourceH = this.video.videoHeight;
    if (!sourceW || !sourceH) return null;
    const scale = Math.min(1, SNAPSHOT_MAX_WIDTH / sourceW);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(sourceW * scale);
    canvas.height = Math.round(sourceH * scale);
    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    this.frame = { width: canvas.width, height: canvas.height };
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', SNAPSHOT_QUALITY));
  }

  /** Draw face boxes returned by the API for the last snapshot. Nothing is drawn without API output. */
  drawFaces(faces = []) {
    const rect = this.video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.round(rect.width * dpr);
    this.overlay.height = Math.round(rect.height * dpr);
    const ctx = this.overlay.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!this.frame || !faces.length) return;

    const known = cssToken('--mint', '#82f3d4');
    const unknown = cssToken('--warning', '#f4ce8a');
    const view = { width: rect.width, height: rect.height };

    for (const face of faces) {
      if (!face?.box) continue;
      const { x, y, width, height } = projectBox(face.box, this.frame, view);
      const color = face.matched ? known : unknown;
      const radius = Math.min(12, width / 4, height / 4);

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
      ctx.stroke();

      // Corner accents.
      ctx.lineWidth = 2;
      ctx.globalAlpha = 1;
      const arm = Math.min(20, width / 3, height / 3);
      ctx.beginPath();
      ctx.moveTo(x - 3, y - 3 + arm); ctx.lineTo(x - 3, y - 3); ctx.lineTo(x - 3 + arm, y - 3);
      ctx.moveTo(x + width + 3 - arm, y + height + 3); ctx.lineTo(x + width + 3, y + height + 3); ctx.lineTo(x + width + 3, y + height + 3 - arm);
      ctx.stroke();

      const label = (face.matched ? face.display_name || 'KNOWN' : 'UNKNOWN').toUpperCase();
      ctx.font = '600 9px Inter, ui-sans-serif, system-ui, sans-serif';
      const textW = ctx.measureText(label).width;
      const tagW = textW + 18;
      const tagH = 20;
      const tagY = y + height + 8;
      ctx.fillStyle = 'rgba(5,10,11,.84)';
      ctx.beginPath();
      ctx.roundRect(x, tagY, tagW, tagH, tagH / 2);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = face.matched ? '#eef7f6' : color;
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x + 9, tagY + tagH / 2 + 0.5);
    }
  }

  clearOverlay() {
    const ctx = this.overlay.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
  }
}
