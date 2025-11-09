# chat_rag_enhanced_voice_local_ai.py
import sys, os, time, struct, sqlite3, logging, re, traceback, json, requests
import speech_recognition as sr, pyttsx3, sqlite_vec
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
AI_TOOLKIT_BASE_URL = "http://127.0.0.1:5272/v1/"
AI_TOOLKIT_MODEL    = "deepseek-r1-distill-qwen-1.5b-cpu-int4-rtn-block-32-acc-level-4"
AI_TEMPERATURE      = 0.2
AI_MAX_TOKENS       = 140

DB_FILE     = "chat_memory.db"
VEC_TABLE   = "messages_vec"
META_TABLE  = "messages_meta"
EMB_MODEL   = "all-MiniLM-L6-v2"
EMB_DIMS    = 384
TOP_K       = 5
MAX_SNIPPET = 240

PHRASE_TIME_LIMIT       = 7
AMBIENT_NOISE_DURATION  = 0.4
LOG_FILE = "chat_memory.log"

# ============================================================
# LOGGING — same as before, no buffering issues
# ============================================================
fmt = "%(asctime)s [%(levelname)s] %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
log = logging.getLogger("AVA-RAG")
log.setLevel(logging.INFO)

# clear previous handlers (VSCode re-runs can double-log)
for h in list(log.handlers):
    log.removeHandler(h)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

log.addHandler(file_handler)
log.addHandler(console_handler)
log.propagate = False
log.info(f"🧾 Logging active → {os.path.abspath(LOG_FILE)}")

# ============================================================
# HELPERS
# ============================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
def is_non_english(s: str): return bool(_CJK_RE.search(s))

def strip_think(text: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if t: return t
    inside = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return inside[-1].strip() if inside else text.strip()

def sanitize_for_tts(text: str) -> str:
    t = strip_think(text)
    t = _CJK_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t:
        t = "Okay."
    return t

# ============================================================
# EMBEDDINGS
# ============================================================
_model = None
def get_model():
    global _model
    if _model is None:
        t0 = time.perf_counter()
        log.info(f"🧠 Loading embedding model: {EMB_MODEL}")
        _model = SentenceTransformer(EMB_MODEL)
        log.info(f"✅ Embedding model ready in {(time.perf_counter()-t0)*1000:.1f} ms")
    return _model

def embed(text: str):
    t0 = time.perf_counter()
    vec = get_model().encode(text, normalize_embeddings=True).tolist()
    log.info(f"🔢 Embedded text ({len(text)} chars) → {len(vec)} dims in {(time.perf_counter()-t0)*1000:.1f} ms")
    return vec

def serialize_f32(vec):
    return struct.pack("%sf" % len(vec), *vec)

# ============================================================
# DATABASE
# ============================================================
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0(embedding float[{EMB_DIMS}])")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {META_TABLE}(
        rowid INTEGER PRIMARY KEY,
        ts REAL NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL
    )""")
    conn.commit()
    meta_rows = conn.execute(f"SELECT COUNT(*) FROM {META_TABLE}").fetchone()[0]
    vec_rows  = conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
    conn.close()
    log.info(f"🗃️ Database initialized and verified. meta_rows={meta_rows}, vec_rows={vec_rows}")

def db_connect():
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def remember(conn, role, text):
    vec = embed(text)
    cur = conn.execute(f"INSERT INTO {VEC_TABLE}(embedding) VALUES (?)", (serialize_f32(vec),))
    rid = cur.lastrowid
    conn.execute(f"INSERT INTO {META_TABLE}(rowid,ts,role,text) VALUES (?,?,?,?)",
                 (rid, time.time(), role, text))
    conn.commit()
    snippet = (text[:MAX_SNIPPET] + "…") if len(text) > MAX_SNIPPET else text
    log.info(f"💾 Remembered [{role}] id={rid}, chars={len(text)}, :: {snippet!r}")

def recall(conn, query, k=TOP_K):
    qv = embed(query)
    rows = conn.execute(
        f"SELECT rowid,distance FROM {VEC_TABLE} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (serialize_f32(qv), k),
    ).fetchall()
    if not rows:
        log.info("🔍 Recall search: hits=0")
        return []
    ids = [rid for rid, _ in rows]
    meta = conn.execute(f"SELECT rowid,role,text FROM {META_TABLE} WHERE rowid IN (%s)" %
                        ",".join(map(str, ids))).fetchall()
    lookup = {r: (ro, tx) for r, ro, tx in meta}
    results = []
    for rid, dist in rows:
        if rid in lookup:
            ro, tx = lookup[rid]
            tx_clean = strip_think(tx)
            results.append({"role": ro, "text": tx_clean})
    log.info(f"🔍 Recall hits: {len(results)}")
    return results

# ============================================================
# LLM CALL
# ============================================================
def local_llm_generate(model, messages):
    url = f"{AI_TOOLKIT_BASE_URL}chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": AI_TEMPERATURE,
        "max_tokens": AI_MAX_TOKENS,
        "stop": ["</think>", "用户说：", "用户:"],
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        log.info(f"✅ LLM responded in {(time.perf_counter()-t0)*1000:.1f} ms")
        log.info("🧾 Raw LLM response:\n" + text[:1200])
        return text
    except Exception as e:
        log.error(f"❌ LLM call failed: {e}\n{traceback.format_exc()}")
        return "(Error: local model unavailable)"

# ============================================================
# PROMPT
# ============================================================
def build_messages(memory_items, user_text):
    system_prompt = (
        "You are called Ava a personal assistant. Reply in English only. "
        "Do NOT introduce yourself. Do NOT mention company. "
        "Be concise, friendly, contextual, and logical."
    )
    ctx = []
    for m in memory_items[:4]:
        t = m["text"].strip()
        if is_non_english(t):
            continue
        if len(t) > 220:
            t = t[:220] + "…"
        ctx.append(f"{m['role']}: {t}")
    if not ctx:
        ctx.append("(no prior context)")
    log.info("🧩 Sending RAG context to model:")
    for l in ctx:
        log.info("   " + l)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(ctx) + f"\n\nUser: {user_text}"}
    ]

# ============================================================
# SPEECH
# ============================================================
def speak(text: str):
    spoken = sanitize_for_tts(text)
    log.info("🔊 Spoken output:\n" + spoken)
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 185)
        engine.setProperty("volume", 1.0)
        engine.say(spoken)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        log.error(f"TTS error: {e}\n{traceback.format_exc()}")

# ============================================================
# MAIN LOOP
# ============================================================
def listen_and_respond():
    log.info(f"🚀 Starting chat_rag_enhanced_voice_local_ai on Python {sys.version}")
    ensure_db()
    r = sr.Recognizer()
    with sr.Microphone() as s:
        log.info("🎙️ Calibrating microphone...")
        r.adjust_for_ambient_noise(s, duration=AMBIENT_NOISE_DURATION)
        print("Ready. Speak when you hear 'Listening...'")
        while True:
            try:
                print("Listening...")
                t0 = time.perf_counter()
                audio = r.listen(s, timeout=None, phrase_time_limit=PHRASE_TIME_LIMIT)
                asr_ms = (time.perf_counter() - t0) * 1000
                try:
                    text = r.recognize_google(audio)
                except sr.UnknownValueError:
                    log.warning("⚠️ ASR: Speech not recognized.")
                    continue
                log.info(f"🎤 ASR recognized: {text!r}")
                log.info(f"⏱️ ASR time: {asr_ms:.1f} ms")

                conn = db_connect()
                memory_items = recall(conn, text, TOP_K)
                messages = build_messages(memory_items, text)
                raw = local_llm_generate(AI_TOOLKIT_MODEL, messages)
                remember(conn, "user", text)
                remember(conn, "assistant", raw)
                speak(raw)
                conn.close()
            except KeyboardInterrupt:
                log.info("🛑 User interrupted. Exiting.")
                break
            except Exception as e:
                log.error(f"❌ Unexpected error: {e}\n{traceback.format_exc()}")

# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    listen_and_respond()
