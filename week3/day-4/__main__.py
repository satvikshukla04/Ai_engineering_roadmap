import subprocess
import sys
from pathlib import Path

# Re-exec with the shared week3 venv Python if fastapi isn't importable
try:
    import uvicorn  # noqa: F401
except ModuleNotFoundError:
    # Reuse day-3/venv which has all week3 packages
    venv_python = Path(__file__).parent.parent / "day-3" / "venv" / "bin" / "python3"
    if venv_python.exists():
        raise SystemExit(subprocess.call([str(venv_python), str(Path(__file__).parent)]))
    sys.exit("No venv found. Run: python3 -m venv week3/day-3/venv && ./week3/day-3/venv/bin/pip install fastapi uvicorn 'pydantic[email]'")

from validates_api import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
