import subprocess
import os
import json
import cv2
from PIL import Image
import torch
from model import Net
from utils import ConfigL, ConfigS, download_weights

class VideoCaptioner:
    def __init__(self, video_path, model_size='L', checkpoint='model.pt'):
        self.video_path = video_path
        self.model_size = model_size
        self.checkpoint = checkpoint
        self.config = ConfigL() if model_size.upper() == 'L' else ConfigS()
        self.setup_model()

    def setup_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Net(
            clip_model=self.config.clip_model,
            text_model=self.config.text_model,
            ep_len=self.config.ep_len,
            num_layers=self.config.num_layers,
            n_heads=self.config.n_heads,
            forward_expansion=self.config.forward_expansion,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
            device=self.device,
        )
        ckp_path = os.path.join(self.config.weights_dir, self.checkpoint)
        if not os.path.isfile(ckp_path):
            download_weights(ckp_path, self.model_size)
        checkpoint = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model.to(self.device)

    def extract_frames(self):
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frames_dir = f"{video_name}_captioning_frames"
        os.makedirs(frames_dir, exist_ok=True)
        command = [
            "ffmpeg", "-i", self.video_path, "-vf", "fps=1/5", # Adjust `fps=1/5` to control frame rate
            os.path.join(frames_dir, "frame_%03d.jpg")
        ]
        subprocess.run(command, check=True)
        return frames_dir

    def generate_captions(self, frames_dir):
        captions = {}
        for frame_file in sorted(os.listdir(frames_dir)):
            frame_path = os.path.join(frames_dir, frame_file)
            img = Image.open(frame_path).convert("RGB")
            img_tensor = self.config.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                caption, _ = self.model(img_tensor)
            captions[frame_file] = caption
        return captions

    def save_captions_to_json(self, captions, frames_dir):
        json_path = os.path.join(frames_dir, "captions.json")
        with open(json_path, "w") as json_file:
            json.dump(captions, json_file, indent=4)

    def caption_video(self):
        frames_dir = self.extract_frames()
        captions = self.generate_captions(frames_dir)
        self.save_captions_to_json(captions, frames_dir)
        print(f"Captions saved to {os.path.join(frames_dir, 'captions.json')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--video-path", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()
    captioner = VideoCaptioner(args.video_path)
    captioner.caption_video()

