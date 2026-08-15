#!/bin/bash
set -e

# Support overriding the command
if [ "$1" = "serve" ] || [ "$1" = "rtsp" ] || [ "$1" = "infer" ]; then
    exec python -m container_id "$@"
else
    # Fallback to serving
    if [ $# -eq 0 ]; then
        exec python -m container_id serve --models /opt/container-id/models --host 0.0.0.0 --port 8000
    else
        exec "$@"
    fi
fi
