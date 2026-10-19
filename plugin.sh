#!/bin/sh
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
sys="$(uname -s)"
case "$sys" in
MINGW* | MSYS* | CYGWIN*) rel="x86_64-pc-windows-msvc/claude-sub.exe" ;;
Darwin*)
    case "$(uname -m)" in
    arm64) rel="aarch64-apple-darwin/claude-sub" ;;
    *) rel="x86_64-apple-darwin/claude-sub" ;;
    esac ;;
*)
    case "$(uname -m)" in
    aarch64 | arm64) rel="aarch64-unknown-linux-musl/claude-sub" ;;
    *) rel="x86_64-unknown-linux-musl/claude-sub" ;;
    esac ;;
esac
if [ -x "$here/$rel" ]; then exec "$here/$rel" "$@"; fi
for cand in "$here/target/release/claude-sub.exe" "$here/target/release/claude-sub" "$here/target/debug/claude-sub.exe" "$here/target/debug/claude-sub"; do
    if [ -x "$cand" ]; then exec "$cand" "$@"; fi
done
echo "claude-sub binary not found in $here" >&2
exit 1
