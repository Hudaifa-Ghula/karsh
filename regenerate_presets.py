import os
import json
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

def greedy_assignment(source_pixels, target_pixels, sidelen, color_weight=255, spatial_weight=1):
    n_pixels = sidelen * sidelen
    
    # 1. Create a matrix of (X, Y) coordinates for both target and source
    coords = np.zeros((n_pixels, 2), dtype=np.float32)
    for i in range(n_pixels):
        coords[i] = (i % sidelen, i // sidelen)
        
    print("Calculating distances and assigning greedily...")
    
    # We will compute distances in batches so we don't blow up memory, 
    # and assign the best available match greedily.
    assignments = [-1] * n_pixels
    available_sources = np.ones(n_pixels, dtype=bool)
    
    for t_idx in range(n_pixels):
        if t_idx % 1000 == 0:
            print(f"Assigned {t_idx}/{n_pixels} pixels...")
            
        t_loc = coords[t_idx]
        t_color = target_pixels[t_idx]
        
        # Spatial distance (distance squared)
        spatial_dist = np.sum((coords - t_loc)**2, axis=1) * spatial_weight
        
        # Color distance (distance squared)
        color_dist = np.sum((source_pixels - t_color)**2, axis=1) * color_weight
        
        total_dist = spatial_dist + color_dist
        
        # We only want to pick from available sources, so we set assigned ones to infinity
        total_dist[~available_sources] = np.inf
        
        # Find the best source match
        best_s_idx = np.argmin(total_dist)
        
        # Assign it
        assignments[t_idx] = int(best_s_idx)
        available_sources[best_s_idx] = False
        
    return assignments

from PIL import Image, ImageOps

def process_presets():
    target_path = "d:/AppPerWeek/tammat/transformify_web/target.png"
    sidelen = 128
    
    # Load and process target
    print(f"Loading target: {target_path}")
    target_img = Image.open(target_path).convert('RGB')
    
    # Target: resize directly to fill 128x128 (this is the desired output shape)
    target_img = target_img.resize((sidelen, sidelen), Image.LANCZOS)
    target_data = np.array(target_img).reshape(-1, 3).astype(np.float32)
    
    presets = ["colorful", "shrek", "Eiffel", "code"]
    base_dir = "d:/AppPerWeek/tammat/transformify_web/presets"
    
    for preset in presets:
        print(f"\nProcessing preset: {preset}")
        p_path = os.path.join(base_dir, preset)
        
        # We need to make sure the folder exists
        os.makedirs(p_path, exist_ok=True)
        
        # Load source
        source_path = os.path.join(p_path, "source.png")
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} not found! Skipping...")
            continue
            
        source_img = Image.open(source_path).convert('RGB')
        # Source: resize directly (no cropping/padding) to match upload behavior
        source_img = source_img.resize((sidelen, sidelen), Image.LANCZOS)
        source_data = np.array(source_img).reshape(-1, 3).astype(np.float32)
        
        # Save exact resized target and output representations
        target_img.save(os.path.join(p_path, "target.png"))
        
        print("Running greedy matching...")
        assignments = greedy_assignment(source_data, target_data, sidelen)
        
        print("Saving assignments...")
        with open(os.path.join(p_path, "assignments.json"), "w") as f:
            json.dump(assignments, f)
            
        print("Saving preview output...")
        output_pixels = np.zeros((sidelen * sidelen, 3), dtype=np.uint8)
        for t_idx, s_idx in enumerate(assignments):
            output_pixels[t_idx] = source_data[s_idx]
            
        output_img = Image.fromarray(output_pixels.reshape(sidelen, sidelen, 3))
        output_img.save(os.path.join(p_path, "output.png"))

if __name__ == "__main__":
    process_presets()
