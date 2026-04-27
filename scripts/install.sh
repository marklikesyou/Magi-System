#!/usr/bin/env sh
set -eu

REPO_URL="${MAGI_REPO_URL:-https://github.com/marklikesyou/Magi-System.git}"
INSTALL_REF="${MAGI_INSTALL_REF:-master}"
INSTALL_SPEC="${MAGI_INSTALL_SPEC:-magi-system[openai,google] @ git+${REPO_URL}@${INSTALL_REF}}"

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

append_tool_path() {
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
  esac
  case ":$PATH:" in
    *":$HOME/.cargo/bin:"*) ;;
    *) PATH="$HOME/.cargo/bin:$PATH" ;;
  esac
  export PATH
}

command -v curl >/dev/null 2>&1 || die "curl is required to install MAGI."
append_tool_path

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  append_tool_path
fi

command -v uv >/dev/null 2>&1 || die "uv installed, but it is not available on PATH."

log "Installing MAGI..."
uv tool install --force "$INSTALL_SPEC"
append_tool_path

command -v magi >/dev/null 2>&1 || die "magi installed, but it is not available on PATH. Add $HOME/.local/bin to PATH."

if magi setup --check >/dev/null 2>&1; then
  log "MAGI is installed and provider setup is ready."
  exit 0
fi

if [ -r /dev/tty ] && [ -w /dev/tty ]; then
  log "MAGI is installed. Configure an AI provider key now."
  magi setup < /dev/tty || die "MAGI installed, but provider setup did not complete. Run: magi setup"
else
  log "MAGI is installed."
  log "Before using it, run: magi setup"
fi
