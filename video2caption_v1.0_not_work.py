import cv2
import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import Net
from utils import ConfigL, ConfigS, download_weights

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-C", "--checkpoint-name", type=str, default="model.pt", help="Checkpoint name")
parser.add_argument("-S", "--size", type=str, default="L", help="Model size [S, L]", choices=["S", "L", "s", "l"])
parser.add_argument("-V", "--video-path", type=str, default="", help="Path to the video file")
parser.add_argument("-R", "--res-path", type=str, default="./data/result/prediction", help="Path to the results folder")
args = parser.parse_args()

# Configuration based on model size
config = ConfigL() if args.size.upper() == "L" else ConfigS()

# Setup environment
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoints and model setup
if not os.path.exists(config.weights_dir):
    os.makedirs(config.weights_dir)

ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)
if not os.path.isfile(ckp_path):
    download_weights(ckp_path, args.size)

# Load model
model = Net(clip_model=config.clip_model, text_model=config.text_model, ep_len=config.ep_len, 
            num_layers=config.num_layers, n_heads=config.n_heads, forward_expansion=config.forward_expansion, 
            dropout=config.dropout, max_len=config.max_len, device=device)
checkpoint = torch.load(ckp_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# Video processing
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    raise ValueError("Error opening video file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frame_count / fps
interval = frame_count // 10  # Adjust this value for different number of frames

# Prepare output directory
if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

# SRT File setup
srt_path = os.path.join(args.res_path, os.path.basename(args.video_path).split('.')[0] + '.srt')
with open(srt_path, 'w') as srt_file:
    for i in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame
        img_path = os.path.join(args.res_path, f'frame_{i}.jpg')
        cv2.imwrite(img_path, frame)
        img = Image.open(img_path).convert('RGB')

        # Generate caption
        with torch.no_grad():
            caption, _ = model(img, temperature=1.0)

        # Write SRT entry
        start_time = i * interval / fps
        end_time = (i + 1) * interval / fps if i < 9 else duration
        srt_file.write(f'{i + 1}\n')
        srt_file.write(f'{cv2_to_srt_time(start_time)} --> {cv2_to_srt_time(end_time)}\n')
        srt_file.write(f'{caption}\n\n')

        # Clean up
        plt.clf()
        plt.close()

def cv2_to_srt_time(cv2_time):
    """ Convert time in seconds to SRT time format """
    hours, remainder = divmod(cv2_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int((seconds-int(seconds))*1000):03}'

cap.release()


