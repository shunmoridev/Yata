class YataPcmProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.seq = 0;
    this.sampleStart = 0;
    this.buffer = new Float32Array(4096);
    this.used = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === "stop") {
        if (this.used > 0) {
          this.emit(this.buffer.slice(0, this.used));
          this.used = 0;
        }
        this.active = false;
        this.port.postMessage({ type: "stopped" });
      }
    };
  }
  emit(samples) {
    const start = this.sampleStart;
    const count = samples.length;
    this.port.postMessage({ seq: this.seq++, sampleStart: start, pcm: samples.buffer }, [samples.buffer]);
    // Transferring samples.buffer detaches it, making samples.length zero.
    this.sampleStart += count;
  }
  process(inputs) {
    if (!this.active) return false;
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    const length = channels[0].length;
    for (let i = 0; i < length; i++) {
      let sample = 0;
      for (let channel = 0; channel < channels.length; channel++) sample += channels[channel][i] || 0;
      this.buffer[this.used++] = sample / channels.length;
      if (this.used === this.buffer.length) {
        const chunk = this.buffer;
        this.emit(chunk);
        this.buffer = new Float32Array(4096);
        this.used = 0;
      }
    }
    return true;
  }
}
registerProcessor("yata-pcm", YataPcmProcessor);
