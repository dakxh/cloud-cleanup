import os
import json
import subprocess
import shutil
import glob

def process_job():
    print("Starting ProPainter Temporal Pipeline...")
    
    if not os.path.exists("manifest.json"):
        raise FileNotFoundError("manifest.json not found.")
        
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
        
    target_video = manifest["target_video"]
    fps = manifest.get("fps", 30.0)
    process_range = manifest.get("processing_range", [0, manifest.get("total_frames", 0)])
    
    os.makedirs("raw_frames", exist_ok=True)
    os.makedirs("output_frames", exist_ok=True)

    # 1. Extract Frames & Audio
    print(f"Extracting frames from {target_video}...")
    subprocess.run(["ffmpeg", "-y", "-i", target_video, "-start_number", "0", "raw_frames/frame_%04d.png"], check=True)
    
    has_audio = False
    try:
        subprocess.run(["ffmpeg", "-y", "-i", target_video, "-vn", "-c:a", "aac", "source_audio.aac"], check=True, stderr=subprocess.DEVNULL)
        if os.path.exists("source_audio.aac"): has_audio = True
    except subprocess.CalledProcessError:
        pass

    # 2. Stage Frames for ProPainter
    # ProPainter expects a folder of frames and a folder of matching masks.
    print(f"Isolating targeted frames: {process_range[0]} to {process_range[1]}")
    pp_frames_dir = "../ProPainter/inputs/target_seq/frames"
    pp_masks_dir = "../ProPainter/inputs/target_seq/masks"
    os.makedirs(pp_frames_dir, exist_ok=True)
    os.makedirs(pp_masks_dir, exist_ok=True)

    total_frames = manifest.get("total_frames", len(os.listdir("raw_frames")))
    
    for i in range(total_frames):
        raw = f"raw_frames/frame_{i:04d}.png"
        out = f"output_frames/frame_{i:04d}.png"
        mask = f"masks/mask_{i:04d}.png"
        
        # If outside the range, bypass the AI to save hours of compute
        if i < process_range[0] or i > process_range[1] or not os.path.exists(mask):
            shutil.copy2(raw, out)
        else:
            # Copy to ProPainter's staging area
            shutil.copy2(raw, os.path.join(pp_frames_dir, f"{i:05d}.png"))
            shutil.copy2(mask, os.path.join(pp_masks_dir, f"{i:05d}.png"))

    # 3. Execute ProPainter Inference
    print("Executing ProPainter Spatial-Temporal Inpainting...")
    print("WARNING: This will take significant time on a CPU.")
    
    # We constrain the sub_video_length to prevent the GitHub runner from running out of memory.
    cmd = [
        "python", "../ProPainter/inference_propainter.py",
        "--video", pp_frames_dir,
        "--mask", pp_masks_dir,
        "--output", "../ProPainter/results",
        "--sub_video_length", "10", # CRITICAL: Keeps RAM usage under the 7GB GitHub limit
        "--fp16" # Attempts half-precision to speed up processing
    ]
    subprocess.run(cmd, check=True)

    # 4. Retrieve Healed Frames
    print("Integrating healed frames back into the timeline...")
    healed_dir = "../ProPainter/results/target_seq/out"
    
    if os.path.exists(healed_dir):
        for healed_frame in glob.glob(os.path.join(healed_dir, "*.png")):
            filename = os.path.basename(healed_frame)
            idx = int(filename.split('.')[0])
            shutil.copy2(healed_frame, f"output_frames/frame_{idx:04d}.png")

    # 5. Mux Final Video
    print("Muxing final high-fidelity video...")
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-start_number", "0", "-i", "output_frames/frame_%04d.png"]
    if has_audio: cmd.extend(["-i", "source_audio.aac"])
    cmd.extend(["-c:v", "libx264", "-crf", "15", "-preset", "fast", "-pix_fmt", "yuv420p"])
    if has_audio: cmd.extend(["-c:a", "copy", "-shortest"])
    cmd.append("final_healed.mp4")
    
    subprocess.run(cmd, check=True)
    print("ProPainter Pipeline Complete.")

if __name__ == "__main__":
    process_job()
