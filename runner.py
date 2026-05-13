import os
import json
import subprocess
import shutil

def process_job():
    print("Starting Spot Healing Render Pipeline...")
    
    # 1. Read the Blueprint
    if not os.path.exists("manifest.json"):
        raise FileNotFoundError("manifest.json not found in the payload.")
        
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
        
    target_video = manifest["target_video"]
    fps = manifest.get("fps", 30.0)
    process_range = manifest.get("processing_range", [0, manifest.get("total_frames", 0)])
    
    # Setup directories
    os.makedirs("raw_frames", exist_ok=True)
    os.makedirs("output_frames", exist_ok=True)

    # 2. Extract Frames & Audio
    print(f"Extracting frames from {target_video}...")
    subprocess.run(["ffmpeg", "-y", "-i", target_video, "-start_number", "0", "raw_frames/frame_%04d.png"], check=True)
    
    print("Extracting original audio track...")
    has_audio = False
    try:
        subprocess.run(["ffmpeg", "-y", "-i", target_video, "-vn", "-c:a", "aac", "source_audio.aac"], check=True, stderr=subprocess.DEVNULL)
        if os.path.exists("source_audio.aac"):
            has_audio = True
    except subprocess.CalledProcessError:
        print("No audio track found or extraction failed. Proceeding without audio.")

    # 3. The AI Processing Loop
    print(f"Processing frames within range: {process_range[0]} to {process_range[1]}")
    total_frames = manifest.get("total_frames", len(os.listdir("raw_frames")))
    
    for i in range(total_frames):
        raw_frame_path = f"raw_frames/frame_{i:04d}.png"
        out_frame_path = f"output_frames/frame_{i:04d}.png"
        mask_path = f"masks/mask_{i:04d}.png"
        
        # If the frame is outside the tracked range, or has no mask, just copy it to save compute
        if i < process_range[0] or i > process_range[1] or not os.path.exists(mask_path):
            shutil.copy2(raw_frame_path, out_frame_path)
            continue
            
        print(f"Applying AI Inpainting to frame {i}...")
        # ---------------------------------------------------------------------
        # AI INTEGRATION POINT
        # Here is where you drop in your chosen Video Inpainting model inference.
        # Example (Pseudo-code):
        # 
        # from your_ai_library import run_temporal_inpaint
        # run_temporal_inpaint(image_path=raw_frame_path, mask_path=mask_path, output_path=out_frame_path)
        # 
        # For now, we simulate the output by just copying the original frame.
        # ---------------------------------------------------------------------
        shutil.copy2(raw_frame_path, out_frame_path)

    # 4. Final HLS/MP4 Compilation
    print("Muxing final video...")
    cmd = [
        "ffmpeg", "-y", 
        "-framerate", str(fps), 
        "-start_number", "0", 
        "-i", "output_frames/frame_%04d.png"
    ]
    
    if has_audio:
        cmd.extend(["-i", "source_audio.aac"])
        
    cmd.extend([
        "-c:v", "libx264", 
        "-crf", "15", 
        "-preset", "fast", 
        "-pix_fmt", "yuv420p"
    ])
    
    if has_audio:
        cmd.extend(["-c:a", "copy", "-shortest"])
        
    cmd.append("final_healed.mp4")
    
    subprocess.run(cmd, check=True)
    print("Pipeline Complete: final_healed.mp4 generated.")

if __name__ == "__main__":
    process_job()
