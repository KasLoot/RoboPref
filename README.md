# RoboPref

RoboPref v2 is a from-scratch rewrite of the preference-aware robotic
task-orchestration prototype.

Status: pre-alpha development (`2.0.0.dev0`).

```bash
# vLLM server model initialization
# First, run the following command to set up an SSH tunnel to the vLLM server:
ssh -N -L 8000:127.0.0.1:8000 \
  -p 10008 -i ~/.ssh/id_ed25519 \
  root@103.196.86.153
```

## Webcam streaming

Stream camera 0 to a browser on this machine:

```powershell
uv sync
uv run python webcam_stream.py
```

Open <http://127.0.0.1:1234>. Press `Ctrl+C` to stop the server.

Useful options:

```powershell
# Find an available camera index
uv run python webcam_stream.py --list-cameras

# Select camera 1 and use Windows DirectShow
uv run python webcam_stream.py --camera 1 --backend dshow

# Make the stream available on the local network (no authentication)
uv run python webcam_stream.py --host 0.0.0.0
```
