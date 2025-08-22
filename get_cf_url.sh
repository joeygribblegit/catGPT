#!/bin/bash

# Simple script to get Cloudflare tunnel URL

if [ "$1" = "start" ]; then
    echo "Starting Cloudflare tunnel and waiting for URL..."
    echo "----------------------------------------"
    
    # Start cloudflared in background and capture output
    cloudflared tunnel --url http://localhost:5050 2>&1 | tee /tmp/cloudflared_output.log &
    CLOUDFLARED_PID=$!
    
    # Wait a moment for the tunnel to start
    sleep 3
    
    # Look for the URL in the output
    echo "Looking for tunnel URL..."
    for i in {1..10}; do
        if grep -q 'https://.*\.trycloudflare\.com' /tmp/cloudflared_output.log; then
            echo ""
            echo "🎯 TUNNEL URL FOUND:"
            grep -o 'https://[^[:space:]]*\.trycloudflare\.com' /tmp/cloudflared_output.log | tail -1
            echo ""
            echo "Press Ctrl+C to stop the tunnel"
            wait $CLOUDFLARED_PID
            break
        fi
        echo "Waiting for URL... ($i/10)"
        sleep 2
    done
    
    # Cleanup
    kill $CLOUDFLARED_PID 2>/dev/null
    rm -f /tmp/cloudflared_output.log
else
    echo "Usage: $0 start"
    echo "This will start cloudflared and extract the tunnel URL"
fi
