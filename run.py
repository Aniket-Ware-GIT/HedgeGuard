import subprocess
import sys
from pathlib import Path

project_dir = Path(__file__).parent
app_path = project_dir / "app.py"

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app_path),
    "--server.headless=false"
])
