import os, json, base64, asyncio, time
import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import websockets
from dotenv import load_dotenv
from datetime import datetime
import pytz
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress audioop deprecation warning for Python 3.12
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioop")
try:
    import audioop
except ImportError:
    raise SystemExit("audioop not available (Python 3.13+). Use Python 3.12.x or add a resampler lib.")


# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
OPENAI_VOICE = os.getenv("OPENAI_VOICE", "alloy")
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")
GREETING_TEXT = os.getenv("GREETING_TEXT", "Hi, this is the Gribble's pet sitting assistant. What's up?")

# Audio settings
TWILIO_RATE_HZ = 8000
OPENAI_PCM_RATE_HZ = 24000
BYTES_PER_SAMPLE = 2

# Voice activity detection
RMS_SPEECH = int(os.getenv("RMS_SPEECH", "1200"))
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "900"))
MAX_UTTER_MS = int(os.getenv("MAX_UTTER_MS", "6000"))
MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "400"))

# Turn-taking settings
CHUNK_MS = 20
STARTUP_GRACE_MS = int(os.getenv("STARTUP_GRACE_MS", "1500"))
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "500"))
MIN_COMMIT_MS = 120
MIN_COMMIT_BYTES_PCM24 = int(OPENAI_PCM_RATE_HZ * (MIN_COMMIT_MS / 1000.0) * BYTES_PER_SAMPLE)

# Security
PHONE_WHITELIST = [phone.strip() for phone in os.getenv("PHONE_WHITELIST", "").split(",") if phone.strip()]

app = FastAPI()
call_info_store = {}

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect><Stream url="wss://{host}/media"/></Connect>
</Response>
"""

def get_pacific_time():
    """Get current time in Pacific timezone."""
    pacific_tz = pytz.timezone('US/Pacific')
    return datetime.now(pacific_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

def format_duration(ms):
    """Format duration in milliseconds to human readable format."""
    seconds = ms / 1000
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def is_phone_whitelisted(phone_number):
    """Check if a phone number is in the whitelist."""
    if not PHONE_WHITELIST:
        logger.warning("No phone whitelist configured - allowing all calls")
        return True
    
    normalized_phone = phone_number.replace("+1", "").replace(" ", "").replace("-", "")
    
    for whitelisted in PHONE_WHITELIST:
        normalized_whitelisted = whitelisted.replace("+", "").replace(" ", "").replace("-", "")
        if normalized_phone == normalized_whitelisted:
            return True
    return False

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host")
    
    # Extract phone number information from Twilio request
    form_data = await request.form()
    from_number = form_data.get("From", "Unknown")
    to_number = form_data.get("To", "Unknown")
    call_sid = form_data.get("CallSid", "Unknown")
    
    # Check if phone number is whitelisted
    if not is_phone_whitelisted(from_number):
        pacific_time = get_pacific_time()
        logger.warning(f"[{pacific_time}] UNAUTHORIZED CALL BLOCKED - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
        
        # Return rejection TwiML
        rejection_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Sorry, this number is not authorized to access this service. Goodbye.</Say>
  <Hangup/>
</Response>"""
        return PlainTextResponse(rejection_xml, media_type="text/xml")
    
    # Store call information for WebSocket retrieval
    call_info_store[call_sid] = {
        "from_number": from_number,
        "to_number": to_number,
        "call_sid": call_sid
    }
    
    # Log call connection with Pacific time (debug level)
    pacific_time = get_pacific_time()
    logger.debug(f"[{pacific_time}] CALL CONNECTED - From: {from_number}, To: {to_number}, CallSid: {call_sid}")
    
    xml = TWIML.format(host=host)
    return PlainTextResponse(xml, media_type="text/xml")


def load_system_prompt():
    """Load system prompt from file, with fallback to default."""
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            logger.warning(f"System prompt file '{SYSTEM_PROMPT_FILE}' not found, using default prompt")
            return "You are a concise, friendly voice assistant to answer any questions about the Gribble household."
    except Exception as e:
        logger.error(f"Error reading system prompt file: {e}, using default prompt")
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

    to_pcm24k, to_ulaw8k = make_converters()

    stream_sid = None
    awaiting_reply = False
    have_speech = False
    turn_started_ms = 0.0
    last_speech_ms = 0.0
    bytes_since_commit = 0
    call_start_ms = 0
    speech_ms_accum = 0
    greeted = False
    call_info = {}


    async def downlink():
        """OpenAI → Twilio audio."""
        try:
            async for msg in ws_ai:
                evt = json.loads(msg)
                et = evt.get("type")

                if et in ("response.output_audio.delta", "output_audio.delta", "response.audio.delta"):
                    audio_b64 = evt.get("delta") or evt.get("audio")
                    if not audio_b64 or not stream_sid:
                        continue
                    payload = to_ulaw8k(base64.b64decode(audio_b64))
                    await ws_twilio.send_text(twilio_media(stream_sid, payload))

                elif et == "error":
                    logger.error(f"OpenAI error: {evt}")

                elif et in ("response.completed", "response.refused", "response.done"):
                    nonlocal awaiting_reply, have_speech
                    awaiting_reply = False
                    have_speech = False

        except Exception as e:
            logger.error(f"OpenAI downlink closed: {e}")
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
                
                call_sid = data.get("start", {}).get("callSid", "Unknown")
                stored_info = call_info_store.get(call_sid, {})
                
                call_info = {
                    "stream_sid": stream_sid,
                    "start_time": call_start_ms,
                    "from_number": stored_info.get("from_number", "Unknown"),
                    "to_number": stored_info.get("to_number", "Unknown"),
                    "call_sid": call_sid
                }
                
                pacific_time = get_pacific_time()
                logger.info(f"[{pacific_time}] CALL STARTED - From: {call_info['from_number']}, To: {call_info['to_number']}, CallSid: {call_info['call_sid']}")
                
                if not greeted:
                    greeted = True
                    logger.info(f"AI greeting queued: {GREETING_TEXT!r}")
                    try:
                        awaiting_reply = True
                        await ws_ai.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["audio", "text"],
                                "instructions": f"SAY THIS EXACTLY (verbatim, no additions): {GREETING_TEXT}",
                                "max_output_tokens": MAX_MODEL_TOKENS
                            }
                        }))
                    except Exception as e:
                        logger.error(f"Failed to send AI greeting: {e}")
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
                # Log call disconnection with Pacific time and duration
                if call_info:
                    call_end_ms = time.time() * 1000
                    call_duration_ms = call_end_ms - call_info["start_time"]
                    pacific_time = get_pacific_time()
                    duration_str = format_duration(call_duration_ms)
                    logger.info(f"[{pacific_time}] CALL DISCONNECTED - From: {call_info['from_number']}, To: {call_info['to_number']}, CallSid: {call_info['call_sid']}, Duration: {duration_str}")
                    call_info_store.pop(call_info['call_sid'], None)
                break

    except WebSocketDisconnect:
        # Log call disconnection due to WebSocket disconnect
        if call_info:
            call_end_ms = time.time() * 1000
            call_duration_ms = call_end_ms - call_info["start_time"]
            pacific_time = get_pacific_time()
            duration_str = format_duration(call_duration_ms)
            logger.info(f"[{pacific_time}] CALL DISCONNECTED (WebSocket) - From: {call_info['from_number']}, To: {call_info['to_number']}, CallSid: {call_info['call_sid']}, Duration: {duration_str}")
            call_info_store.pop(call_info['call_sid'], None)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
