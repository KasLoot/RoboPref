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