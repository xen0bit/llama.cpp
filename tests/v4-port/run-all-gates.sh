#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/gate-loader.sh"
NGL=0   "$DIR/gate-coherence.sh"
NGL=999 "$DIR/gate-coherence.sh"
"$DIR/gate-speed.sh"
"$DIR/gate-tools.sh"
"$DIR/gate-server-chat.sh"
"$DIR/gate-server-chat-q8.sh"
MODE=warn-fa-off "$DIR/gate-server-chat-q8.sh"
"$DIR/gate-imatrix.sh"
echo "ALL GATES PASS"
