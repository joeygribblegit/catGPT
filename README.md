# Twilio OpenAI Voice Assistant

A real-time voice assistant built with Twilio, OpenAI's Realtime API, and FastAPI. Perfect for creating intelligent voice interfaces for phone calls.

## Features

- 🎤 **Real-time voice conversations** with OpenAI's GPT models
- 📞 **Twilio phone integration** for incoming/outgoing calls
- 🌐 **Cloudflare tunnel** for secure webhook delivery
- 🎯 **Customizable VAD** (Voice Activity Detection) for noise filtering
- 📝 **System prompt customization** via external file
- 🎨 **Multiple voice options** (alloy, aria, verse, shimmer, coral)

## Quick Start

### Prerequisites

- Python 3.12.x (required for audioop support)
- Twilio account with phone number
- OpenAI API key
- Cloudflare account (for tunneling)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd twilio_openai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn websockets python-dotenv
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. **Create system prompt**
   ```bash
   # Edit system_prompt.txt with your assistant's personality
   ```

### Configuration

#### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - VAD Settings
RMS_SPEECH=2000          # Speech detection threshold (higher = less sensitive)
END_SILENCE_MS=1200      # Silence duration before stopping
MIN_SPEECH_MS=800        # Minimum speech duration to trigger
MAX_MODEL_TOKENS=800     # Maximum response length

# Voice Settings
OPENAI_VOICE=alloy       # alloy, aria, verse, shimmer, coral
OPENAI_REALTIME_MODEL=gpt-4o-mini-realtime-preview

# Greeting Settings
USE_AI_GREETING=true
GREETING_TEXT="Hi, this is my assistant. How can I help?"
```

#### System Prompt (system_prompt.txt)

Create a `system_prompt.txt` file with your assistant's personality and instructions.

### Running the Server

1. **Start the server**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 5050 --log-level info
   ```

2. **Set up Cloudflare tunnel**
   ```bash
   cloudflared tunnel --url http://localhost:5050
   ```

3. **Configure Twilio webhook**
   - Set your Twilio phone number's webhook to: `https://your-tunnel-url.ngrok.io/voice`

### Development Workflow

The project includes helpful aliases (add to your `.bashrc`):

```bash
# Tmux sessions
alias server='tmux attach-session -t api'      # Connect to server session
alias cf='tmux attach-session -t cloudflare'   # Connect to cloudflare session

# Server management
alias server_start='uvicorn server:app --host 0.0.0.0 --port 5050 --log-level info'
alias cf_start='./get_cf_url.sh start'         # Start tunnel with URL extraction
alias cf_restart='sudo systemctl restart cloudflared.service'

# Environment
alias reload_env='source .env'                  # Reload environment variables
alias restart_server='pkill -f "uvicorn server:app" && sleep 2 && server_start'
```

## VAD Tuning

Adjust Voice Activity Detection sensitivity in your `.env`:

- **Too sensitive to background noise?** Increase `RMS_SPEECH` (2000-3000)
- **Not responsive enough?** Decrease `RMS_SPEECH` (1500-1800)
- **Cutting off too quickly?** Increase `END_SILENCE_MS` (1200-1500)
- **Missing short words?** Decrease `MIN_SPEECH_MS` (600-800)

## Project Structure

```
twilio_openai/
├── server.py              # Main FastAPI server
├── system_prompt.txt      # AI assistant personality
├── get_cf_url.sh         # Cloudflare tunnel URL extractor
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Audio not working**: Ensure Python 3.12.x (audioop requirement)
2. **High latency**: Check network connection and OpenAI API status
3. **False triggers**: Adjust VAD settings in `.env`
4. **Cut-off responses**: Increase `MAX_MODEL_TOKENS`

### Logs

- Server logs: Check uvicorn output
- Twilio logs: Check Twilio console
- OpenAI logs: Check API usage dashboard

## Security Notes

- Never commit `.env` files (contains API keys)
- Use HTTPS for production webhooks
- Regularly rotate API keys
- Monitor API usage and costs

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
