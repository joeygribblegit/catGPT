import os, json, base64, asyncio, time
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import websockets
from dotenv import load_dotenv
load_dotenv()

# ⚠️ Use Python 3.12.x or earlier. If you're on 3.13+, install a resampler lib (e.g., "samplerate")
try:
    import audioop  # stdlib until 3.12
except ImportError:
    raise SystemExit("audioop not available (Python 3.13+). Use Python 3.12.x or add a resampler lib.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # set in your environment or .env
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")  # alloy, aria, verse, shimmer, coral...

# System prompt configuration
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# Greeting options
USE_TWILIO_GREETING = os.getenv("USE_TWILIO_GREETING", "false").lower() == "true"
USE_AI_GREETING = os.getenv("USE_AI_GREETING", "false").lower() == "true"
GREETING_TEXT = os.getenv("GREETING_TEXT", "Hi, this is the Gribble's pet sitting assistant. What's up?")

# Audio pipeline: Twilio 8k μ-law  <->  OpenAI PCM16 ~24k
TWILIO_RATE_HZ = 8000
OPENAI_PCM_RATE_HZ = 24000
BYTES_PER_SAMPLE = 2

# VAD / turn-taking knobs
RMS_SPEECH = int(os.getenv("RMS_SPEECH", "1200"))
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "900"))
MAX_UTTER_MS = int(os.getenv("MAX_UTTER_MS", "6000"))
MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "400"))

# Anti–double-greeting / empty-commit guards
CHUNK_MS = 20
STARTUP_GRACE_MS = int(os.getenv("STARTUP_GRACE_MS", "1500"))
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "500"))
MIN_COMMIT_MS = 120
MIN_COMMIT_BYTES_PCM24 = int(OPENAI_PCM_RATE_HZ * (MIN_COMMIT_MS / 1000.0) * BYTES_PER_SAMPLE)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

# TwiML: include <Say> only when USE_TWILIO_GREETING=true
if USE_TWILIO_GREETING:
    TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{greeting}</Say>
  <Connect><Stream url="wss://{host}/media"/></Connect>
</Response>
"""
else:
    TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect><Stream url="wss://{host}/media"/></Connect>
</Response>
"""

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host")
    if USE_TWILIO_GREETING:
        xml = TWIML.format(host=host, greeting=GREETING_TEXT)
    else:
        xml = TWIML.format(host=host)
    return PlainTextResponse(xml, media_type="text/xml")


def load_system_prompt():
    """Load system prompt from file, with fallback to default."""
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            print(f"System prompt file '{SYSTEM_PROMPT_FILE}' not found, using default prompt")
            return "You are a concise, friendly voice assistant to answer any questions about the Gribble household."
    except Exception as e:
        print(f"Error reading system prompt file: {e}, using default prompt")
        return "You are a concise, friendly voice assistant to answer any questions about the Gribble household."


async def openai_connect():
    """Create a Realtime WS session with explicit PCM16 I/O and pinned English transcription."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Set it in your environment or .env file.")
    
    # Load system prompt from file
    system_prompt = load_system_prompt()
    
    uri = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
    ws = await websockets.connect(
        uri, extra_headers=headers, ping_interval=20, ping_timeout=20, max_size=10_000_000
    )
    session_update = {
        "type": "session.update",
        "session": {
            "voice": OPENAI_VOICE,
            "modalities": ["text", "audio"],
            "instructions": system_prompt,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "max_response_output_tokens": MAX_MODEL_TOKENS,
            # Pin ASR to English to avoid random language flips
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": "en"
            }
        },
    }
    await ws.send(json.dumps(session_update))
    return ws


def make_converters():
    """Stateful resamplers: 8k μ-law <-> 24k PCM16."""
    state_up = None
    state_down = None

    def ulaw8k_to_pcm24k(b64_ulaw: str) -> bytes:
        nonlocal state_up
        mulaw = base64.b64decode(b64_ulaw)
        pcm8k = audioop.ulaw2lin(mulaw, BYTES_PER_SAMPLE)
        pcm24k, state_up = audioop.ratecv(
            pcm8k, BYTES_PER_SAMPLE, 1, TWILIO_RATE_HZ, OPENAI_PCM_RATE_HZ, state_up
        )
        return pcm24k

    def pcm24k_to_ulaw8k(pcm24k: bytes) -> str:
        nonlocal state_down
        pcm8k, state_down = audioop.ratecv(
            pcm24k, BYTES_PER_SAMPLE, 1, OPENAI_PCM_RATE_HZ, TWILIO_RATE_HZ, state_down
        )
        mulaw = audioop.lin2ulaw(pcm8k, BYTES_PER_SAMPLE)
        return base64.b64encode(mulaw).decode("ascii")

    return ulaw8k_to_pcm24k, pcm24k_to_ulaw8k


def twilio_media(stream_sid: str, b64_payload: str) -> str:
    return json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": b64_payload}})

async def send_text_turn(ws_ai, text: str):
    await ws_ai.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text}
            ]
        }
    }))
    await ws_ai.send(json.dumps({"type": "response.create"}))

@app.websocket("/media")
async def media(ws_twilio: WebSocket):
    await ws_twilio.accept()
    ws_ai = await openai_connect()

    # Per-connection converter state
    to_pcm24k, to_ulaw8k = make_converters()

    stream_sid = None
    awaiting_reply = False
    have_speech = False
    turn_started_ms = 0.0
    last_speech_ms = 0.0

    # anti-empty-commit / anti-early-commit trackers
    bytes_since_commit = 0
    call_start_ms = 0
    speech_ms_accum = 0

    greeted = False


    async def downlink():
        """OpenAI → Twilio audio."""
        try:
            async for msg in ws_ai:
                evt = json.loads(msg)
                et = evt.get("type")
                # print("OPENAI EVENT:", et)

                if et in ("response.output_audio.delta", "output_audio.delta", "response.audio.delta"):
                    audio_b64 = evt.get("delta") or evt.get("audio")
                    if not audio_b64 or not stream_sid:
                        continue
                    payload = to_ulaw8k(base64.b64decode(audio_b64))
                    await ws_twilio.send_text(twilio_media(stream_sid, payload))

                elif et == "error":
                    print("OPENAI ERROR:", evt)

                elif et in ("response.completed", "response.refused", "response.done"):
                    nonlocal awaiting_reply, have_speech
                    awaiting_reply = False
                    have_speech = False

        except Exception as e:
            print("OPENAI downlink closed:", e)
            try:
                await ws_ai.close()
            except:
                pass

    task_downlink = asyncio.create_task(downlink())

    try:
        async for text in ws_twilio.iter_text():
            data = json.loads(text)
            ev = data.get("event")

            if ev == "start":
                stream_sid = data["start"]["streamSid"]
                call_start_ms = time.time() * 1000
                # Optional: AI greeting in the model's own voice
                if USE_AI_GREETING and not USE_TWILIO_GREETING and not greeted:
                    greeted = True
                    print(f"AI greeting (verbatim) queued: {GREETING_TEXT!r}")
                    try:
                        awaiting_reply = True  # block VAD commits during greeting
                        await ws_ai.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["audio", "text"],
                                "instructions": f"SAY THIS EXACTLY (verbatim, no additions): {GREETING_TEXT}",
                                "max_output_tokens": MAX_MODEL_TOKENS
                            }
                        }))
                    except Exception as e:
                        print("Failed to send AI greeting:", e)
                        awaiting_reply = False

            elif ev == "media":
                # 20ms μ-law chunk from caller
                b64_ulaw = data["media"]["payload"]

                # Convert and push into OpenAI
                pcm24 = to_pcm24k(b64_ulaw)
                await ws_ai.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24).decode("ascii")
                }))
                bytes_since_commit += len(pcm24)

                # Simple RMS VAD to find utterance boundaries
                rms = audioop.rms(pcm24, BYTES_PER_SAMPLE)
                now_ms = time.time() * 1000

                if not awaiting_reply:
                    if rms >= RMS_SPEECH:
                        if not have_speech:
                            have_speech = True
                            turn_started_ms = now_ms
                        last_speech_ms = now_ms
                        # accumulate approximate speech time by chunk size
                        speech_ms_accum = min(speech_ms_accum + CHUNK_MS, MAX_UTTER_MS)

                turn_len = (now_ms - turn_started_ms) if have_speech else 0
                sil_len  = (now_ms - last_speech_ms) if have_speech else 0

                # gates to prevent early/noisy commits
                eligible_time   = (now_ms - call_start_ms) >= STARTUP_GRACE_MS
                eligible_speech = speech_ms_accum >= MIN_SPEECH_MS
                enough_bytes    = bytes_since_commit >= MIN_COMMIT_BYTES_PCM24

                should_end = have_speech and not awaiting_reply and eligible_time and eligible_speech and (
                    sil_len >= END_SILENCE_MS or turn_len >= MAX_UTTER_MS
                )

                if should_end and enough_bytes:
                    await ws_ai.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await ws_ai.send(json.dumps({"type": "response.create"}))
                    awaiting_reply = True
                    # reset counters after a real commit
                    speech_ms_accum = 0
                    bytes_since_commit = 0

            elif ev == "stop":
                break

    except WebSocketDisconnect:
        pass
    finally:
        # One last chance to flush, but only if there is real buffered audio
        try:
            enough = bytes_since_commit >= MIN_COMMIT_BYTES_PCM24
            if enough and not awaiting_reply:
                await ws_ai.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await ws_ai.send(json.dumps({"type": "response.create"}))
        except Exception:
            pass
        try:
            await ws_ai.close()
        except Exception:
            pass
        task_downlink.cancel()


