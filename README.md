# chat_rag_enhanced_voice_local_ai
chat_rag_enhanced_voice_local_ai — Offline, voice-enabled RAG assistant with on-device LLMs, local embeddings, and FishAudio TTS for natural speech. Private, fast, and fully local — no cloud required.


pip install -r requirements.txt

# Test the LLM end point
curl http://127.0.0.1:5272/v1/models

# chat_rag_enhanced_voice_local_ai

## 🧠 Description
`chat_rag_enhanced_voice_local_ai` — **Offline, voice-enabled RAG assistant** with on-device LLMs, local embeddings, and **Kokoro-TTS** for natural speech synthesis.  
Private, fast, and completely local — **no internet or cloud access required**.

This project combines:
- 🎤 Offline speech recognition (via `SpeechRecognition` + `PyAudio`)
- 🧬 On-device embeddings (`SentenceTransformers` + `sqlite-vec`)
- 💬 Local reasoning with a running LLM endpoint (via `AI Toolkit` or similar)
- 🔊 High-fidelity, offline text-to-speech using **Kokoro-TTS**
- 🧾 Persistent chat memory stored locally in SQLite

---

## ⚙️ System Requirements

| Component | Description |
|------------|-------------|
| **Python** | 3.10 or newer (tested on 3.11.9) |
| **CUDA (optional)** | Recommended for GPU acceleration |
| **Microphone** | Required for speech input |
| **Speakers** | Required for TTS playback |

---

## 🧩 Installation

### 1️⃣ Install PyTorch manually (GPU-optimized)
> Run these commands **before** installing the main dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

---

### 2️⃣ Install requirements

```bash
pip install -r requirements.txt
```

### ✅ `requirements.txt`
```txt
torch
soundfile
kokoro
SpeechRecognition
sentence-transformers
sqlite-vec
pyaudio
```

---

## 🧠 Verifying Local LLM Endpoint

Before running the assistant, verify that your **local AI model service** is active and responding:

```bash
curl http://127.0.0.1:5272/v1/models
```

If the service responds with model details, you’re ready to go.

---

## 🚀 Running the Assistant

To start the voice-enabled offline assistant:

```bash
python chat_rag_enhanced_voice_local_ai.py
```

### What happens:
1. Initializes logging and SQLite memory.
2. Calibrates microphone and listens for your voice.
3. Transcribes your speech locally.
4. Performs RAG recall and sends context to the local LLM.
5. Generates an intelligent reply.
6. Synthesizes and plays natural speech using **Kokoro-TTS**.

---

## 🎧 Kokoro-TTS Notes

- The first time you run it, **Kokoro** automatically downloads its speech model weights (cached locally afterward).
- Default voice: `af_heart` (American English)
- Output file: `output.wav`
- Sample rate: `24,000 Hz`
- Compatible with both CPU and CUDA GPUs.

---

## 🗃️ Local Data & Logs

| File | Purpose |
|------|----------|
| `chat_memory.db` | Stores conversation history and embeddings |
| `chat_memory.log` | Runtime log file (for debugging or review) |
| `output.wav` | Latest synthesized speech output |

All data stays on your local machine.  
No external network calls are made (aside from the local model endpoint).

---

## 🧹 Troubleshooting

| Issue | Solution |
|--------|-----------|
| **No microphone detected** | Ensure `pyaudio` is installed and working. |
| **ASR not recognizing speech** | Increase ambient noise duration or speak closer to mic. |
| **Kokoro error: missing dependency** | Verify `soundfile` and `torch` are installed correctly. |
| **No audio playback** | Check default system audio device. |

---

## 🧰 Development Notes

- Built for **offline operation**
- Integrates seamlessly with **local AI Toolkit** models like:
  - `deepseek-r1-distill-qwen-1.5b-cpu-int4-rtn-block-32-acc-level-4`
- Can easily swap to other local model endpoints.

---

## 🪶 License
MIT License © 2025 William Paul Best  
All rights reserved.

---

## ❤️ Credits
- [Kokoro-TTS](https://github.com/hexgrad/Kokoro-82M) — Local neural TTS  
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — Vector search extension  
- [SentenceTransformers](https://www.sbert.net/) — Local embeddings  
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) — Local voice capture
