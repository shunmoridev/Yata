(() => {
  "use strict";
  const t = (key, params) => globalThis.YataI18n.t(key, params);
  const MAX_PENDING_BYTES = 12 * 1024 * 1024;
  const MAX_PENDING_SECONDS = 60;
  const MAX_SOCKET_BUFFERED = 1_048_576;
  const HEADER_BYTES = 12;
  const FATAL_CLOSE_CODES = new Set([4000, 4400, 4403, 4404, 4409]);

  function encodeFrame(seq, sampleStart, pcm) {
    const packet = new ArrayBuffer(HEADER_BYTES + pcm.byteLength);
    const view = new DataView(packet);
    view.setUint32(0, seq >>> 0, true);
    view.setBigUint64(4, BigInt(sampleStart), true);
    new Uint8Array(packet, HEADER_BYTES).set(new Uint8Array(pcm));
    return packet;
  }

  class FrameQueue {
    constructor(sampleRate) {
      this.sampleRate = sampleRate;
      this.frames = [];
      this.bytes = 0;
      this.gap = null;
    }

    push(frame, networkBytes = 0) {
      this.frames.push(frame);
      this.bytes += frame.pcm.byteLength + HEADER_BYTES;
      const maxSamples = this.sampleRate * MAX_PENDING_SECONDS - Math.ceil(networkBytes / 4);
      const end = frame.sampleStart + frame.pcm.byteLength / 4;
      while (this.frames.length &&
        (this.bytes + networkBytes > MAX_PENDING_BYTES || end - this.frames[0].sampleStart > maxSamples)) {
        const dropped = this.frames.shift();
        this.bytes -= dropped.pcm.byteLength + HEADER_BYTES;
        if (!this.gap) this.gap = { start: dropped.sampleStart, end: dropped.sampleStart };
        this.gap.end = dropped.sampleStart + dropped.pcm.byteLength / 4;
      }
    }

    take(bufferedAmount) {
      const out = [];
      let buffered = bufferedAmount;
      while (this.frames.length && buffered + this.frames[0].pcm.byteLength + HEADER_BYTES <= MAX_SOCKET_BUFFERED) {
        const frame = this.frames.shift();
        this.bytes -= frame.pcm.byteLength + HEADER_BYTES;
        if (this.gap) {
          out.push(JSON.stringify({
            type: "audio_gap", start_sample: this.gap.start, end_sample: this.gap.end,
            next_seq: frame.seq, next_sample: frame.sampleStart,
          }));
          this.gap = null;
        }
        const packet = encodeFrame(frame.seq, frame.sampleStart, frame.pcm);
        out.push(packet);
        buffered += packet.byteLength;
      }
      return out;
    }

    undelivered() {
      if (!this.gap && !this.frames.length) return null;
      const first = this.gap ? this.gap.start : this.frames[0].sampleStart;
      const last = this.frames[this.frames.length - 1];
      return { start: first, end: last ? last.sampleStart + last.pcm.byteLength / 4 : this.gap.end };
    }
  }

  function clockMs(session, at) {
    const offset = Number(session?.clock_offset_ms) || 0;
    const started = Date.parse(session?.clock_started_utc || "");
    return Number.isFinite(started) ? offset + Math.max(0, at - started) : offset;
  }

  // Chrome exposes no tab title or URL; a page can opt in to reveal its origin through the Capture Handle
  // API, which is the only URL Yata can learn.
  function describeSurface(stream) {
    const video = stream.getVideoTracks()[0];
    let origin = "";
    try { origin = (video?.getCaptureHandle?.() || {}).origin || ""; } catch { origin = ""; }
    return { url: /^https?:\/\//.test(origin) ? origin : "", surface: video?.getSettings?.().displaySurface || "" };
  }

  class Capture {
    constructor(options) {
      this.sessionId = null;
      this.onState = options.onState || (() => {});
      this.onCaption = options.onCaption || (() => {});
      this.onGap = options.onGap || (() => {});
      this.onStats = options.onStats || (() => {});
      this.stopped = true;
      this.ready = false;
      this.reconnects = 0;
      this.stats = { capture_frames: 0, capture_bytes: 0, sent_frames: 0, sent_bytes: 0 };
    }

    // Called inside the click handler, before any session exists: Chrome requires user activation for
    // getDisplayMedia.
    async open() {
      this.stopped = false;
      this.stream = await navigator.mediaDevices.getDisplayMedia({
        // Video is required by the API; a tiny, slow track keeps the share alive cheaply.
        video: { displaySurface: "browser", frameRate: 1, width: { max: 320 }, height: { max: 180 } },
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: false },
        preferCurrentTab: false,
        selfBrowserSurface: "exclude",
        surfaceSwitching: "include",
        systemAudio: "include",
        windowAudio: "window",
      });
      const audioTracks = this.stream.getAudioTracks();
      if (!audioTracks.length) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stopped = true;
        throw new Error(t("capture.noAudio"));
      }
      this.label = audioTracks[0].label || "";
      for (const track of this.stream.getTracks()) {
        track.addEventListener("ended", () => {
          if (this.stopped) return;
          if (!this.context) { void this.stop(); return; }
          this.onState("share_ended", t("capture.shareEnded"));
          void this.finish();
        }, { once: true });
      }
      return describeSurface(this.stream);
    }

    async begin(session) {
      if (this.stopped || !this.stream) throw new Error(t("capture.stopped"));
      this.sessionId = session.id;
      this.capturedAt = new Date().toISOString();
      this.captureBaseMs = clockMs(session, Date.parse(this.capturedAt));
      const audioTracks = this.stream.getAudioTracks();
      this.context = new AudioContext();
      await this.context.audioWorklet.addModule("/assets/pcm-processor.js");
      await this.context.resume();
      this.queue = new FrameQueue(this.context.sampleRate);
      const source = this.context.createMediaStreamSource(new MediaStream(audioTracks));
      this.node = new AudioWorkletNode(this.context, "yata-pcm", { numberOfInputs: 1, numberOfOutputs: 1 });
      // The shared tab keeps playing on its own; this branch is muted and only keeps the graph running.
      const mute = this.context.createGain();
      mute.gain.value = 0;
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.55;
      source.connect(this.analyser);
      source.connect(this.node);
      this.node.connect(mute);
      mute.connect(this.context.destination);
      this.node.port.onmessage = ({ data }) => {
        if (data.type === "stopped") {
          if (this.workletStopped) this.workletStopped();
          return;
        }
        if (this.stopped || !data.pcm) return;
        this.stats.capture_frames++;
        this.stats.capture_bytes += data.pcm.byteLength;
        this.queue.push(data, this.socketBuffered());
        this.drain();
      };
      this.statsTimer = setInterval(() => this.onStats(this.snapshot()), 1_000);
      this.connect();
      this.onState("starting_audio", t("capture.preparing"));
    }

    socketBuffered() {
      return this.socket && this.socket.readyState === WebSocket.OPEN ? this.socket.bufferedAmount : 0;
    }

    snapshot() {
      return {
        ...this.stats,
        queue_frames: this.queue ? this.queue.frames.length : 0,
        queue_bytes: this.queue ? this.queue.bytes : 0,
        socket_buffered_bytes: this.socketBuffered(),
        sample_rate: this.context ? this.context.sampleRate : 0,
        track_state: this.stream ? (this.stream.getAudioTracks()[0]?.readyState || "unknown") : "none",
        context_state: this.context ? this.context.state : "none",
        socket_state: this.socket ? ["connecting", "open", "closing", "closed"][this.socket.readyState] : "none",
        reconnects: this.reconnects,
        label: this.label || "",
      };
    }

    connect() {
      if (this.stopped) return;
      const ws = new WebSocket(`ws://${location.host}/api/v1/sessions/${this.sessionId}/audio`);
      ws.binaryType = "arraybuffer";
      this.socket = ws;
      this.ready = false;
      ws.onopen = () => {
        if (this.socket !== ws) return;
        ws.send(JSON.stringify({ format: "pcm_f32le", sample_rate: this.context.sampleRate, channels: 1, captured_at_utc: this.capturedAt }));
      };
      ws.onmessage = (event) => {
        if (this.socket !== ws) return;
        let data;
        try { data = JSON.parse(event.data); } catch { return; }
        if (data.type === "asr_ready") {
          this.ready = true;
          this.reconnects = 0;
          this.onState("capturing_audio", t("capture.capturing"));
          this.drain();
        } else if (data.type === "asr_loading") {
          this.onState("starting_audio", t("capture.loading"));
        } else if (data.type === "caption_partial" || data.type === "caption_final" || data.type === "caption_clear") {
          this.onCaption(data);
        } else if (data.type === "flush_requested") {
          void this.finish();
        } else if (data.type === "finish_audio_ready") {
          if (this.finishResolver) this.finishResolver(true);
        } else if (data.type === "asr_error") {
          this.onState("asr_error", data.reason || t("capture.asrError"));
        } else if (data.type === "asr_unavailable" || data.type === "error") {
          this.onState("error", data.reason || t("capture.startFailed"));
          void this.stop();
        } else if (data.type === "session_stopped") {
          void this.stop();
        }
      };
      ws.onclose = (event) => {
        if (this.socket !== ws || this.stopped) return;
        this.ready = false;
        if (FATAL_CLOSE_CODES.has(event.code)) {
          if (event.code !== 4000) this.onState("error", t("capture.refused", { code: event.code }));
          void this.stop();
          return;
        }
        if (this.finishResolver) {
          this.finishResolver(false);
          return;
        }
        this.reconnects++;
        if (this.reconnects > 5) {
          this.onState("error", t("capture.reconnectFailed"));
          void this.stop();
          return;
        }
        this.onState("reconnecting", t("capture.reconnecting"));
        this.reconnectTimer = setTimeout(() => this.connect(), Math.min(8_000, 500 * 2 ** this.reconnects));
      };
    }

    drain() {
      const ws = this.socket;
      if (!this.ready || !ws || ws.readyState !== WebSocket.OPEN || !this.queue) return;
      for (const message of this.queue.take(ws.bufferedAmount)) {
        ws.send(message);
        if (typeof message !== "string") {
          this.stats.sent_frames++;
          this.stats.sent_bytes += message.byteLength - HEADER_BYTES;
        }
      }
      // Backpressure: retry soon if frames are still waiting for the socket buffer to empty.
      if (this.queue.frames.length && !this.drainTimer) {
        this.drainTimer = setTimeout(() => { this.drainTimer = null; this.drain(); }, 100);
      }
    }

    async finish() {
      if (this.finishing) return this.finishing;
      if (this.stopped) return false;
      this.finishing = (async () => {
        if (this.node) {
          await new Promise((resolve) => {
            const timer = setTimeout(resolve, 2_000);
            this.workletStopped = () => { clearTimeout(timer); resolve(); };
            this.node.port.postMessage({ type: "stop" });
          });
        }
        const deadline = Date.now() + 90_000;
        while (Date.now() < deadline && !this.stopped) {
          this.drain();
          if (this.ready && !this.queue.frames.length && this.socket?.readyState === WebSocket.OPEN) break;
          await new Promise((resolve) => setTimeout(resolve, 100));
        }
        let confirmed = false;
        if (this.socket?.readyState === WebSocket.OPEN && !this.stopped) {
          confirmed = await new Promise((resolve) => {
            const timer = setTimeout(() => resolve(false), Math.max(1_000, deadline - Date.now()));
            this.finishResolver = (value) => { clearTimeout(timer); resolve(value); };
            this.socket.send(JSON.stringify({ type: "finish_audio" }));
          });
        }
        await this.stop();
        return confirmed;
      })();
      return this.finishing;
    }

    async stop() {
      if (this.stopped && !this.stream) return;
      this.stopped = true;
      this.ready = false;
      clearTimeout(this.reconnectTimer);
      clearTimeout(this.drainTimer);
      clearInterval(this.statsTimer);
      const lost = this.queue ? this.queue.undelivered() : null;
      if (lost && this.context) {
        const toMs = (sample) => this.captureBaseMs + Math.round(sample * 1000 / this.context.sampleRate);
        this.onGap({ reason: t("capture.unsent"), capture_start_ms: toMs(lost.start), capture_end_ms: toMs(lost.end) });
      }
      this.onStats(this.snapshot());
      const socket = this.socket;
      this.socket = null;
      if (socket && socket.readyState <= WebSocket.OPEN) socket.close(1000, "capture stopped");
      if (this.node) this.node.disconnect();
      if (this.analyser) this.analyser.disconnect();
      this.analyser = null;
      if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
      if (this.context) await this.context.close().catch(() => {});
      this.context = null;
      this.onState("stopped", "");
    }
  }

  globalThis.YataCapture = { Capture, FrameQueue, encodeFrame, clockMs, describeSurface };
})();
