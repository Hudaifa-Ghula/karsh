from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid
import numpy as np
from PIL import Image
import json
import time

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PRESETS_DIR = "presets"
RESULTS_DIR = "results"
TARGET_IMAGE = "target.png"

for d in [UPLOAD_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def greedy_assignment(source_pixels, target_pixels, sidelen, color_weight=255, spatial_weight=1):
    n_pixels = sidelen * sidelen
    y, x = np.mgrid[0:sidelen, 0:sidelen]
    coords = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float32)
    
    assignments = np.full(n_pixels, -1, dtype=np.int32)
    available_sources = np.ones(n_pixels, dtype=bool)
    
    source_pixels = source_pixels.astype(np.float32)
    target_pixels = target_pixels.astype(np.float32)
    
    for t_idx in range(n_pixels):
        t_loc = coords[t_idx]
        t_color = target_pixels[t_idx]
        
        mask = available_sources
        valid_coords = coords[mask]
        valid_colors = source_pixels[mask]
        
        spatial_dist = np.sum((valid_coords - t_loc)**2, axis=1) * spatial_weight
        color_dist = np.sum((valid_colors - t_color)**2, axis=1) * color_weight
        
        total_dist = spatial_dist + color_dist
        local_best = np.argmin(total_dist)
        
        # Map local index back to global index
        global_indices = np.where(mask)[0]
        best_s_idx = global_indices[local_best]
        
        assignments[t_idx] = best_s_idx
        available_sources[best_s_idx] = False
        
    return assignments.tolist()

@app.get("/presets")
async def get_presets():
    presets = []
    if os.path.exists(PRESETS_DIR):
        for d in os.listdir(PRESETS_DIR):
            p_path = os.path.join(PRESETS_DIR, d)
            if os.path.isdir(p_path):
                presets.append({
                    "id": d,
                    "name": d.capitalize(),
                    "has_assignments": os.path.exists(os.path.join(p_path, "assignments.json"))
                })
    return presets

@app.get("/presets/{preset_id}/source")
async def get_preset_source(preset_id: str):
    path = os.path.join(PRESETS_DIR, preset_id, "source.png")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/presets/{preset_id}/target.png")
async def get_preset_target(preset_id: str):
    path = os.path.join(PRESETS_DIR, preset_id, "target.png")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/presets/{preset_id}/assignments")
async def get_preset_assignments(preset_id: str):
    path = os.path.join(PRESETS_DIR, preset_id, "assignments.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.post("/transform")
async def transform(file: UploadFile = File(...), sidelen: int = Form(128)):
    session_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{session_id}_input.png")
    output_path = os.path.join(RESULTS_DIR, f"{session_id}_output.png")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    from PIL import ImageOps
    # Process
    src_img = Image.open(input_path).convert('RGB')
    tgt_img = Image.open(TARGET_IMAGE).convert('RGB')
    
    # Source: resize directly (no cropping/padding)
    src_img = src_img.resize((sidelen, sidelen), Image.LANCZOS)
    # Target: resize directly (this is the desired result shape, no padding)
    tgt_img = tgt_img.resize((sidelen, sidelen), Image.LANCZOS)
    
    src_pixels = np.array(src_img).reshape(-1, 3)
    tgt_pixels = np.array(tgt_img).reshape(-1, 3)
    
    assignments = greedy_assignment(src_pixels, tgt_pixels, sidelen)
    
    return {
        "session_id": session_id,
        "assignments": assignments,
        "source_res": sidelen
    }

# Mount static files if frontend is built
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
