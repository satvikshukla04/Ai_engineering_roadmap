import subprocess
import sys
from pathlib import Path

# Re-exec with the local venv Python if uvicorn isn't importable
try:
    import uvicorn  # noqa: F401
except ModuleNotFoundError:
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
    if venv_python.exists():
        raise SystemExit(subprocess.call([str(venv_python), str(Path(__file__).parent)]))
    sys.exit("uvicorn not found. Run: python3 -m venv day-3/venv && ./day-3/venv/bin/pip install fastapi uvicorn")

from asyncroutes import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
