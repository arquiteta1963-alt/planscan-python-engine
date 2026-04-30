# custom_floorplan_detect (Python)

Serviço de detecção real de paredes em plantas baixas. Pareado ao provider
`custom_floorplan_ai` do frontend Lovable.

## Endpoints
- `GET  /health` → `{ ok: true }`
- `POST /detect` (multipart, campo `file`) → JSON no contrato `FloorPlanDetections`.

## Rodar local
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy (qualquer host com Python 3.11)
- Fly.io, Render, Railway, Cloud Run, etc.
- Use o `Dockerfile` incluso.

## Conectar ao Lovable
Depois do deploy, defina o secret no projeto Lovable:
```
CUSTOM_FLOORPLAN_PY_URL = https://<seu-host>/detect
```
O server route `/api/custom-floorplan-detect` vai proxiar automaticamente.
