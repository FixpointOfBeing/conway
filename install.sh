#!/usr/bin/env bash
#
# Build conway in release mode and install the binary to ~/.local/bin.
#
# Usage:
#   ./install.sh
#   BIN_DIR=/usr/local/bin ./install.sh   # install somewhere else
#
set -euo pipefail

: "${HOME:?HOME is not set}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIN_NAME="conway"
DEST_DIR="${BIN_DIR:-$HOME/.local/bin}"
SRC="$SCRIPT_DIR/target/release/$BIN_NAME"

echo "==> cargo build --release"
cargo build --release

if [[ ! -x "$SRC" ]]; then
    echo "error: build finished but $SRC is missing or not executable" >&2
    exit 1
fi

echo "==> install $SRC -> $DEST_DIR/$BIN_NAME"
mkdir -p "$DEST_DIR"
install -m 755 "$SRC" "$DEST_DIR/$BIN_NAME"

echo "==> installed $DEST_DIR/$BIN_NAME"

case ":$PATH:" in
    *":$DEST_DIR:"*) ;;
    *)
        echo
        echo "note: $DEST_DIR is not in your PATH. Add this to your shell profile:"
        echo "  export PATH=\"$DEST_DIR:\$PATH\""
        ;;
esac
