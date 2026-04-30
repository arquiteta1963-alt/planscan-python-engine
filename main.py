"""
Serviço Python — custom_floorplan_detect
========================================

FastAPI + OpenCV avançado. Recebe uma imagem por multipart/form-data
(campo `file`) e devolve detections no contrato:

{
  "walls":   [ { "x1","y1","x2","y2","thickness","confidence" } ],
  "doors":   [],
  "windows": [],
  "rooms":   [],
  "confidence": float,
  "preview_image": "data:image/png;base64,...",   # opcional
  "warnings": [ ... ],
  "width": int, "height": int
}

Pipeline:
  1. Decodifica imagem.
  2. Tenta corrigir perspectiva (maior contorno quadrilátero ~ folha A4).
  3. Separa canal de paredes (preto) e marcações (vermelho) por HSV.
  4. Remove textos / blobs pequenos (componentes conectados).
  5. Canny + HoughLinesP.
  6. Snap horizontal/vertical, agrupa próximos, mescla colineares.
  7. Devolve walls com x1,y1,x2,y2,thickness,confidence.

NÃO inventa rooms/doors. windows = []. doors = []. rooms = [].
A normalização final / SVG é responsabilidade do frontend
(provider custom_floorplan_ai → detectionsToFloorPlanResult).

Rodar local:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Deploy: Fly.io, Render, Railway, Cloud Run — qualquer host com Python 3.11.
Depois cole a URL pública no secret CUSTOM_FLOORPLAN_PY_URL do projeto Lovable.
"""

from __future__ import annotations

import base64
import io
import math
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="custom_floorplan_detect", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------- Utilidades geométricas ----------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos em TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _try_perspective(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Tenta endireitar a folha. Retorna (img, foi_corrigida)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    page_area = w * h
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > page_area * 0.25:
            quad = _order_quad(approx.reshape(4, 2).astype(np.float32))
            (tl, tr, br, bl) = quad
            wA = np.linalg.norm(br - bl)
            wB = np.linalg.norm(tr - tl)
            hA = np.linalg.norm(tr - br)
            hB = np.linalg.norm(tl - bl)
            maxW = int(max(wA, wB))
            maxH = int(max(hA, hB))
            if maxW < 100 or maxH < 100:
                return img, False
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(quad, dst)
            warped = cv2.warpPerspective(img, M, (maxW, maxH))
            return warped, True
    return img, False


# ---------- Máscaras de cor ----------

def _mask_black_walls(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive — robusto a sombra e iluminação irregular.
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    # Remove ruído pequeno (textos, pontilhado, cotas).
    th = _remove_small_blobs(th, min_area=80)
    # Fecha cantos.
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def _mask_red(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def _remove_small_blobs(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


# ---------- Hough + limpeza geométrica ----------

def _hough_segments(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=30,
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in l[0]) for l in lines]


def _snap_h_v(seg, tol_deg: float = 7.0):
    x1, y1, x2, y2 = seg
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    a = abs(ang) % 180
    if a < tol_deg or a > 180 - tol_deg:
        y = (y1 + y2) // 2
        return (x1, y, x2, y)
    if abs(a - 90) < tol_deg:
        x = (x1 + x2) // 2
        return (x, y1, x, y2)
    return None  # diagonal — descarta


def _length(s):
    return math.hypot(s[2] - s[0], s[3] - s[1])


def _merge_collinear(segs, gap=8, tol=4):
    """Mescla segmentos colineares próximos (mesma orientação H/V)."""
    horiz = [s for s in segs if s[1] == s[3]]
    vert = [s for s in segs if s[0] == s[2]]

    def merge_axis(items, axis_idx):
        # axis_idx 0 = vertical (mesmo x), agrupa por x; 1 = horizontal, agrupa por y
        items = sorted(items, key=lambda s: (s[axis_idx], min(s[0], s[2]), min(s[1], s[3])))
        merged = []
        for s in items:
            placed = False
            for i, m in enumerate(merged):
                if abs(s[axis_idx] - m[axis_idx]) <= tol:
                    if axis_idx == 1:  # horizontal: y igual, intervalo em x
                        a1, a2 = sorted([m[0], m[2]])
                        b1, b2 = sorted([s[0], s[2]])
                        if b1 <= a2 + gap and a1 <= b2 + gap:
                            new_y = (m[axis_idx] + s[axis_idx]) // 2
                            merged[i] = (min(a1, b1), new_y, max(a2, b2), new_y)
                            placed = True
                            break
                    else:  # vertical
                        a1, a2 = sorted([m[1], m[3]])
                        b1, b2 = sorted([s[1], s[3]])
                        if b1 <= a2 + gap and a1 <= b2 + gap:
                            new_x = (m[axis_idx] + s[axis_idx]) // 2
                            merged[i] = (new_x, min(a1, b1), new_x, max(a2, b2))
                            placed = True
                            break
            if not placed:
                merged.append(s)
        return merged

    return merge_axis(horiz, 1) + merge_axis(vert, 0)


def _estimate_thickness(mask: np.ndarray, seg) -> int:
    """Estima espessura amostrando a normal da linha."""
    x1, y1, x2, y2 = seg
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if y1 == y2:  # horizontal → mede vertical
        col = mask[max(0, cy - 20): cy + 20, cx]
    else:  # vertical → mede horizontal
        col = mask[cy, max(0, cx - 20): cx + 20]
    on = int(np.sum(col > 0))
    return max(1, on)


# ---------- Endpoint ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty_file"}, status_code=400)

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "decode_failed"}, status_code=400)

    warnings: List[str] = []

    # 1. Perspectiva
    img2, corrected = _try_perspective(img)
    if not corrected:
        warnings.append("Perspectiva da folha não detectada — usando imagem original.")

    h, w = img2.shape[:2]

    # 2. Máscaras
    walls_mask = _mask_black_walls(img2)
    red_mask = _mask_red(img2)
    # Remove vermelho do canal de paredes (não confunde marcações com parede).
    walls_mask = cv2.bitwise_and(walls_mask, cv2.bitwise_not(red_mask))

    # 3. Hough + snap H/V
    raw_segs = _hough_segments(walls_mask)
    snapped = [s for s in (_snap_h_v(s) for s in raw_segs) if s is not None]

    # 4. Filtra muito curtas
    min_len = max(20, int(0.02 * max(w, h)))
    snapped = [s for s in snapped if _length(s) >= min_len]

    # 5. Mescla colineares
    merged = _merge_collinear(snapped, gap=10, tol=5)

    # 6. Confidence por comprimento normalizado
    diag = math.hypot(w, h)
    walls_out = []
    for s in merged:
        thick = _estimate_thickness(walls_mask, s)
        conf = float(min(1.0, _length(s) / (diag * 0.6)))
        walls_out.append({
            "x1": int(s[0]), "y1": int(s[1]),
            "x2": int(s[2]), "y2": int(s[3]),
            "thickness": int(thick),
            "confidence": round(conf, 3),
        })

    if not walls_out:
        warnings.append("Nenhuma parede confiável detectada. Tente uma imagem com melhor contraste e iluminação uniforme.")

    overall_conf = float(np.mean([w_["confidence"] for w_ in walls_out])) if walls_out else 0.0

    # Preview anotado
    preview = img2.copy()
    for w_ in walls_out:
        cv2.line(preview, (w_["x1"], w_["y1"]), (w_["x2"], w_["y2"]), (0, 0, 0), 3)
    ok, buf = cv2.imencode(".png", preview)
    preview_b64 = ""
    if ok:
        preview_b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "walls": walls_out,
        "doors": [],
        "windows": [],
        "rooms": [],
        "confidence": round(overall_conf, 3),
        "preview_image": preview_b64,
        "warnings": warnings,
        "width": int(w),
        "height": int(h),
    }
