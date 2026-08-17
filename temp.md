New model ssh commands:
Gemma 4
ssh -N -L 8000:127.0.0.1:8000 \
  -p 31653 -i ~/.ssh/id_ed25519 \
  root@198.13.252.112

EmbeddingGemma
ssh -N -L 8080:127.0.0.1:8080 \
  -p 26179 -i ~/.ssh/id_ed25519 \
  root@47.47.180.20

SAM 3.1:
ssh -N -L 9000:127.0.0.1:9000 \
  -p 26179 -i ~/.ssh/id_ed25519 \
  root@47.47.180.20