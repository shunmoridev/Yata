(() => {
  "use strict";
  const BARS = 112;

  function hexToRgb(hex) {
    const value = parseInt(String(hex).replace("#", ""), 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
  }

  class Waveform {
    constructor(canvas, options) {
      const opts = options || {};
      this.canvas = canvas;
      this.ctx = canvas && canvas.getContext ? canvas.getContext("2d") : null;
      this.onLevel = opts.onLevel || (() => {});
      this.analyser = null;
      this.heights = new Float32Array(BARS);
      this.tint = "#0eb2d4";
      this.accent = "#6152f3";
      this.speech = 0; // 0..1, eased toward the target so the glow swells and fades
      this.speechTarget = 0;
      this.frame = 0;
      this.running = false;
      this.reduced = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (this.ctx && typeof ResizeObserver === "function") {
        new ResizeObserver(() => this.resize()).observe(canvas);
      }
      this.resize();
    }

    attach(analyser) {
      this.analyser = analyser || null;
      if (this.analyser) {
        this.freq = new Uint8Array(this.analyser.frequencyBinCount);
        this.time = new Float32Array(this.analyser.fftSize);
      }
    }

    setTint(color, accent) {
      this.tint = color || "#0eb2d4";
      this.accent = accent || "#6152f3";
    }

    setSpeech(active) {
      this.speechTarget = active ? 1 : 0;
    }

    resize() {
      if (!this.ctx) return;
      const ratio = Math.min(2, (typeof devicePixelRatio === "number" ? devicePixelRatio : 1));
      const width = Math.max(1, this.canvas.clientWidth || 600);
      const height = Math.max(1, this.canvas.clientHeight || 200);
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = width;
      this.height = height;
    }

    start() {
      if (!this.ctx || this.running) return;
      this.running = true;
      const step = () => {
        if (!this.running) return;
        this.draw();
        // Reduced motion: a slow refresh keeps the level readable without constant movement.
        if (this.reduced) setTimeout(() => requestAnimationFrame(step), 250);
        else requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    stop() {
      this.running = false;
    }

    targets() {
      const out = new Float32Array(BARS);
      const half = BARS / 2;
      if (this.analyser) {
        this.analyser.getByteFrequencyData(this.freq);
        // Speech lives roughly in the first third of the spectrum at 48 kHz; log-spaced bins.
        const usable = Math.max(8, Math.floor(this.freq.length * 0.36));
        for (let i = 0; i < half; i++) {
          const from = Math.floor(Math.pow(i / half, 1.6) * usable);
          const to = Math.max(from + 1, Math.floor(Math.pow((i + 1) / half, 1.6) * usable));
          let sum = 0;
          for (let k = from; k < to; k++) sum += this.freq[k];
          const value = Math.pow(sum / (to - from) / 255, 1.35);
          out[half - 1 - i] = value;
          out[half + i] = value;
        }
      } else {
        const t = this.frame * 0.035;
        for (let i = 0; i < BARS; i++) {
          const x = (i - half) / half;
          const envelope = Math.exp(-x * x * 3.2);
          out[i] = 0.04 + envelope * (0.1 + 0.08 * Math.sin(t * 2 + i * 0.4) * Math.cos(t - i * 0.17));
        }
      }
      return out;
    }

    level() {
      if (!this.analyser) return 0;
      this.analyser.getFloatTimeDomainData(this.time);
      let sum = 0;
      for (let i = 0; i < this.time.length; i++) sum += this.time[i] * this.time[i];
      const rms = Math.sqrt(sum / this.time.length);
      const db = 20 * Math.log10(rms + 1e-8);
      return Math.max(0, Math.min(1, (db + 60) / 60));
    }

    draw() {
      const ctx = this.ctx;
      const width = this.width;
      const height = this.height;
      this.frame++;
      this.speech += (this.speechTarget - this.speech) * 0.08;
      const targets = this.targets();
      for (let i = 0; i < BARS; i++) {
        const target = targets[i];
        // Fast attack, slow release, like a VU meter.
        this.heights[i] += (target - this.heights[i]) * (target > this.heights[i] ? 0.55 : 0.12);
      }
      const level = this.level();
      this.onLevel(level);
      ctx.clearRect(0, 0, width, height);

      const [r, g, b] = hexToRgb(this.tint);
      const [ar, ag, ab] = hexToRgb(this.accent);
      const mid = height / 2;
      const gap = 2;
      const barWidth = Math.max(1.5, (width * 0.86) / BARS - gap);
      const total = BARS * (barWidth + gap);
      const left = (width - total) / 2;

      const halo = ctx.createRadialGradient(width / 2, mid, 4, width / 2, mid, width * 0.42);
      halo.addColorStop(0, `rgba(${r},${g},${b},${0.10 + this.speech * 0.16})`);
      halo.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = halo;
      ctx.fillRect(0, 0, width, height);

      const gradient = ctx.createLinearGradient(0, mid - height * 0.45, 0, mid + height * 0.45);
      gradient.addColorStop(0, `rgba(${r},${g},${b},0.95)`);
      gradient.addColorStop(0.5, `rgba(${ar},${ag},${ab},1)`);
      gradient.addColorStop(1, `rgba(${r},${g},${b},0.95)`);
      ctx.fillStyle = gradient;
      ctx.shadowColor = `rgba(${r},${g},${b},${0.35 + this.speech * 0.45})`;
      ctx.shadowBlur = 8 + this.speech * 14;
      for (let i = 0; i < BARS; i++) {
        const h = Math.max(2, this.heights[i] * height * 0.86);
        const x = left + i * (barWidth + gap);
        const radius = Math.min(barWidth / 2, 3);
        roundRect(ctx, x, mid - h / 2, barWidth, h, radius);
      }
      ctx.shadowBlur = 0;

      if (this.analyser) {
        ctx.strokeStyle = `rgba(${ar},${ag},${ab},${0.25 + this.speech * 0.35})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        const stride = Math.max(1, Math.floor(this.time.length / width));
        for (let x = 0, i = 0; x < width && i < this.time.length; x++, i += stride) {
          const y = mid + this.time[i] * height * 0.42;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fill();
  }

  globalThis.YataWaveform = { Waveform, BARS };
})();
