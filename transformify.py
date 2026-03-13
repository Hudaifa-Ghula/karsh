import os
import json
import numpy as np
from PIL import Image
import argparse
import time

def greedy_assignment(source_pixels, target_pixels, sidelen, color_weight=255, spatial_weight=1):
    """
    Optimized greedy matching of source pixels to target pixels.
    """
    n_pixels = sidelen * sidelen
    
    # Create coordinate grid
    y, x = np.mgrid[0:sidelen, 0:sidelen]
    coords = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float32)
    
    print(f"Assigning {n_pixels} pixels...")
    
    assignments = np.full(n_pixels, -1, dtype=np.int32)
    available_sources = np.ones(n_pixels, dtype=bool)
    
    # Pre-calculate spatial distances for all pairs is too large (16k * 16k * 4 bytes ~ 1GB)
    # But we can do it row-wise or in batches if needed. 
    # For now, let's keep it memory efficient.
    
    source_pixels = source_pixels.astype(np.float32)
    target_pixels = target_pixels.astype(np.float32)
    
    start_time = time.time()
    
    # Bottleneck: The loop over target pixels.
    # We can speed this up by processing in chunks if we weren't "greedy" (i.e. using KD-Tree),
    # but exact greedy parity with Rust/Presets requires this order or equivalent.
    
    for t_idx in range(n_pixels):
        if t_idx % 2000 == 0 and t_idx > 0:
            elapsed = time.time() - start_time
            eta = (elapsed / t_idx) * (n_pixels - t_idx)
            print(f"Progress: {t_idx}/{n_pixels} ({t_idx/n_pixels:.1%}) - ETA: {eta:.1f}s")
            
        t_loc = coords[t_idx]
        t_color = target_pixels[t_idx]
        
        # Vectorized distance calculation
        # Only compute for available sources
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
        
    print(f"Assignment complete in {time.time() - start_time:.2f}s")
    return assignments.tolist()

def transform_image(source_path, target_path, output_path, sidelen=128):
    print(f"Transforming {source_path} to look like {target_path}")
    
    # Load images
    src_img = Image.open(source_path).convert('RGB')
    tgt_img = Image.open(target_path).convert('RGB')
    
    # Resize to working resolution
    src_img = src_img.resize((sidelen, sidelen), Image.LANCZOS)
    tgt_img = tgt_img.resize((sidelen, sidelen), Image.LANCZOS)
    
    src_pixels = np.array(src_img).reshape(-1, 3)
    tgt_pixels = np.array(tgt_img).reshape(-1, 3)
    
    # Run assignment
    assignments = greedy_assignment(src_pixels, tgt_pixels, sidelen)
    
    # Construct output
    out_pixels = np.zeros((sidelen * sidelen, 3), dtype=np.uint8)
    for t_idx, s_idx in enumerate(assignments):
        out_pixels[t_idx] = src_pixels[s_idx]
        
    out_img = Image.fromarray(out_pixels.reshape(sidelen, sidelen, 3))
    out_img.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformify: Morph pixels from source to target.")
    parser.add_argument("--source", type=str, required=True, help="Path to source image")
    parser.add_argument("--target", type=str, default="d:/AppPerWeek/tammat/result.png", help="Path to target image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save result")
    parser.add_argument("--size", type=int, default=128, help="Resolution (default 128)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.target):
        print(f"Error: Target image {args.target} not found.")
        exit(1)
        
    transform_image(args.source, args.target, args.output, args.size)
