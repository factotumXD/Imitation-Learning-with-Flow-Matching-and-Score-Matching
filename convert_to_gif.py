"""Convert MP4 videos to GIFs using moviepy."""

from pathlib import Path
from moviepy import VideoFileClip

# Input MP4 files
mp4_files = [
    Path(r"d:\book\cs\ai\reinforcement_learning\cs185\homework_spring2026\hw1\exp\mse\wandb\files\media\videos\eval\rollout_ep4_70000_b0ff790b1f431aac2e8d.mp4"),
    Path(r"d:\book\cs\ai\reinforcement_learning\cs185\homework_spring2026\hw1\exp\flow\wandb\files\media\videos\eval\rollout_ep4_70000_78b41094005b5e08bc0a.mp4"),
    Path(r"d:\book\cs\ai\reinforcement_learning\cs185\homework_spring2026\hw1\exp\scoreSigma1\wandb\files\media\videos\eval\rollout_ep4_70000_05c22f7552f34dd36d16.mp4"),
    Path(r"d:\book\cs\ai\reinforcement_learning\cs185\homework_spring2026\hw1\exp\scoreSigma0.1\wandb\files\media\videos\eval\rollout_ep4_70000_89849623fce7bc13abea.mp4"),
]

# Output directory
output_dir = Path(r"d:\book\cs\ai\reinforcement_learning\cs185\homework_spring2026\hw1\gif")
output_dir.mkdir(parents=True, exist_ok=True)

# GIF names
gif_names = [
    "mse_policy.gif",
    "flow_policy.gif",
    "score_sigma1_policy.gif",
    "score_sigma0.1_policy.gif",
]

# Convert each video to GIF
for mp4_path, gif_name in zip(mp4_files, gif_names):
    if not mp4_path.exists():
        print(f"Warning: {mp4_path} not found, skipping...")
        continue
    
    output_path = output_dir / gif_name
    
    # Load video and convert to GIF
    clip = VideoFileClip(str(mp4_path))
    clip.write_gif(str(output_path), fps=10)
    
    print(f"Converted {mp4_path.name} -> {output_path}")

print("Conversion complete!")
