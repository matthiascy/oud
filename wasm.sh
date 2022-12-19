#!/usr/bin/env sh

case $1 in
    build)
        command -v wasm-pack >/dev/null 2>&1 || { echo >&2 "wasm-pack is required but it's not installed. ... Aborting."; exit 1; }
        wasm-pack build --release --target web
        ;;
    serve)
        command -v python3 >/dev/null 2>&1 || { echo >&2 "Python 3's http.server is required but it's not installed. ... Aborting."; exit 1; }
        python3 -m http.server
        ;;
    *)
        echo "Unknown command: $1"
        exit 1
        ;;
esac    
