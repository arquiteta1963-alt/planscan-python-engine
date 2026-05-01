import os
import uuid
import base64
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client


app = FastAPI(title="PlanScan Python Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
def home():
    return {"ok": True, "service": "PlanScan Python Engine"}


@app.get("/health")
def health():
    return {"ok": True}


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def normalize_line(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx >= dy:
        y = int((y1 + y2) / 2)
        return {
            "id": f"wall_{uuid.uuid4().hex[:8]}",
            "x1": int(min(x1, x2)),
            "y1": y,
            "x2": int(max(x1, x2)),
            "y2": y,
            "orientation": "horizontal",
            "type": "wall",
            "confidence": 0.65,
        }
    else:
        x = int((x1 + x2) / 2)
        return {
            "id": f"wall_{uuid.uuid4().hex[:8]}",
            "x1": x,
            "y1": int(min(y1, y2)),
            "x2": x,
            "y2": int(max(y1, y2)),
            "orientation": "vertical",
            "type": "wall",
            "confidence": 0.65,
        }


def detect_walls(image: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=35,
        maxLineGap=18,
    )

    walls = []

    if lines is None:
        return walls

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 35:
            continue

        wall = normalize_line(x1, y1, x2, y2)
        walls.append(wall)

    return walls


def merge_walls(walls: List[Dict[str, Any]], tolerance: int = 14) -> List[Dict[str, Any]]:
    horizontal = [w for w in walls if w["orientation"] == "horizontal"]
    vertical = [w for w in walls if w["orientation"] == "vertical"]

    merged = []

    def merge_group(group, orientation):
        used = [False] * len(group)
        result = []

        for i, wall in enumerate(group):
            if used[i]:
                continue

            current = wall.copy()
            used[i] = True

            changed = True
            while changed:
                changed = False

                for j, other in enumerate(group):
                    if used[j]:
                        continue

                    if orientation == "horizontal":
                        same_axis = abs(current["y1"] - other["y1"]) <= tolerance
                        touching = not (
                            other["x2"] < current["x1"] - tolerance
                            or other["x1"] > current["x2"] + tolerance
                        )

                        if same_axis and touching:
                            current["x1"] = min(current["x1"], other["x1"])
                            current["x2"] = max(current["x2"], other["x2"])
                            current["y1"] = current["y2"] = int(
                                (current["y1"] + other["y1"]) / 2
                            )
                            used[j] = True
                            changed = True

                    else:
                        same_axis = abs(current["x1"] - other["x1"]) <= tolerance
                        touching = not (
                            other["y2"] < current["y1"] - tolerance
                            or other["y1"] > current["y2"] + tolerance
                        )

                        if same_axis and touching:
                            current["y1"] = min(current["y1"], other["y1"])
                            current["y2"] = max(current["y2"], other["y2"])
                            current["x1"] = current["x2"] = int(
                                (current["x1"] + other["x1"]) / 2
                            )
                            used[j] = True
                            changed = True

            current["id"] = f"wall_{uuid.uuid4().hex[:8]}"
            current["confidence"] = 0.78
            result.append(current)

        return result

    merged.extend(merge_group(horizontal, "horizontal"))
    merged.extend(merge_group(vertical, "vertical"))

    return merged


def build_svg(width: int, height: int, walls: List[Dict[str, Any]]) -> str:
    lines = []

    for wall in walls:
        lines.append(
            f'<line x1="{wall["x1"]}" y1="{wall["y1"]}" '
            f'x2="{wall["x2"]}" y2="{wall["y2"]}" '
            f'stroke="#111827" stroke-width="4" stroke-linecap="square" />'
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect width="100%" height="100%" fill="#ffffff"/>
      {''.join(lines)}
    </svg>
    """

    return svg.strip()


def save_to_supabase(image_url: str, walls: List[Dict[str, Any]], metadata: Dict[str, Any]):
    if not supabase:
        return {"saved": False, "reason": "Supabase not configured"}

    try:
        result = (
            supabase.table("training_samples")
            .insert(
                {
                    "image_url": image_url,
                    "ai_walls": walls,
                    "metadata": metadata,
                }
            )
            .execute()
        )

        return {"saved": True, "result": str(result)}
    except Exception as e:
        return {"saved": False, "reason": str(e)}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    content = await file.read()

    np_arr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "status": "error",
            "message": "Imagem inválida ou não suportada",
        }

    height, width = image.shape[:2]

    raw_walls = detect_walls(image)
    clean_walls = merge_walls(raw_walls)

    svg = build_svg(width, height, clean_walls)

    confidence = 0
    if len(raw_walls) > 0:
        confidence = min(95, int((len(clean_walls) / max(len(raw_walls), 1)) * 100))

    response = {
        "status": "success",
        "provider": "custom_floorplan_ai",
        "width": width,
        "height": height,
        "walls": clean_walls,
        "doors": [],
        "windows": [],
        "rooms": [],
        "svg": svg,
        "preview_image": image_to_base64(image),
        "confidence": confidence,
        "geometryStats": {
            "raw_walls": len(raw_walls),
            "clean_walls": len(clean_walls),
            "doors": 0,
            "windows": 0,
            "rooms": 0,
        },
        "warnings": [
            "Engine OpenCV inicial: detecta e consolida paredes, mas ainda não reconhece ambientes automaticamente."
        ],
    }

    save_result = save_to_supabase(
        image_url=file.filename or "uploaded_image",
        walls=clean_walls,
        metadata={
            "provider": "custom_floorplan_ai",
            "width": width,
            "height": height,
            "raw_walls": len(raw_walls),
            "clean_walls": len(clean_walls),
            "confidence": confidence,
        },
    )

    response["supabase"] = save_result

    return response
