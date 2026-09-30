// Runs Whisper off the main thread so the page stays responsive.
// Tries the graphics card (WebGPU) first, which is much faster, then falls back to the CPU.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

let asr = null, loaded = "";
const post = (m) => self.postMessage(m);
const progress_callback = (p) => { if (p.status === "progress") post({ type: "load", pct: Math.round(p.progress) }); };

async function load(model, gpu) {
  const key = model + (gpu ? ":gpu" : ":cpu");
  if (loaded === key) return;
  if (gpu) {
    asr = await pipeline("automatic-speech-recognition", model, {
      device: "webgpu", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" }, progress_callback });
  } else {
    try { asr = await pipeline("automatic-speech-recognition", model, { dtype: "q8", progress_callback }); }
    catch { asr = await pipeline("automatic-speech-recognition", model, { progress_callback }); }
  }
  loaded = key;
}

async function run(audio, language) {
  const opts = { return_timestamps: "word", task: "transcribe", chunk_length_s: 30, stride_length_s: 5 };
  if (language) opts.language = language;
  return (await asr(audio, opts)).chunks || [];
}

self.onmessage = async ({ data: { audio, model, language } }) => {
  try {
    let gpu = !!self.navigator?.gpu, chunks;
    if (gpu) {
      try { await load(model, true); post({ type: "run", gpu: true }); chunks = await run(audio, language); }
      catch { gpu = false; loaded = ""; }
    }
    if (!gpu) { await load(model, false); post({ type: "run", gpu: false }); chunks = await run(audio, language); }
    post({ type: "done", chunks });
  } catch (e) { post({ type: "error", message: e.message || String(e) }); }
};
