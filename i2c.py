import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import Net
from utils import ConfigL, ConfigS, download_weights

class ImageCaptioner:
    def __init__(self, model_size='L', checkpoint_name='model.pt', res_path='./data/result/prediction', temperature=1.0):
        self.model_size = model_size
        self.checkpoint_name = checkpoint_name
        self.res_path = res_path
        self.temperature = temperature
        self.config = ConfigL() if model_size.upper() == 'L' else ConfigS()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.img_path = None  # img_path is not required at initialization

    def load_model(self):
        ckp_path = os.path.join(self.config.weights_dir, self.checkpoint_name)
        if not os.path.isfile(ckp_path):
            if not os.path.exists(self.config.weights_dir):
                os.makedirs(self.config.weights_dir)
            download_weights(ckp_path, self.model_size)
        
        model = Net(
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
        checkpoint = torch.load(ckp_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def set_image_path(self, img_path):
        self.img_path = img_path

    def generate_caption(self, save_image=True):
        assert self.img_path is not None and os.path.isfile(self.img_path), "Image path is not set or the image does not exist"

        img = Image.open(self.img_path).convert("RGB")
        with torch.no_grad():
            caption, _ = self.model(img, self.temperature)

        if save_image:
            try:
                self.save_captioned_image(img, caption)
            except Exception as e:
                print("Error in save captioned images: ", str(e))


        return caption

    def save_captioned_image(self, img, caption):
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)

        plt.imshow(img)
        plt.title(caption)
        plt.axis("off")
        img_save_path = (
            f'{os.path.split(self.img_path)[-1].split(".")[0]}-R{self.model_size.upper()}.jpg'
        )
        plt.savefig(os.path.join(self.res_path, img_save_path), bbox_inches="tight")
        plt.clf()
        plt.close()
        print(f"Image saved to {os.path.join(self.res_path, img_save_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument("-I", "--img-path", type=str, required=True, help="Path to the image")
    parser.add_argument("-S", "--size", type=str, default="L", help="Model size [S, L]")
    parser.add_argument("-C", "--checkpoint-name", type=str, default="model.pt", help="Checkpoint name")
    parser.add_argument("-R", "--res-path", type=str, default="./data/result/prediction", help="Path to the results folder")
    parser.add_argument("-T", "--temperature", type=float, default=1.0, help="Temperature for sampling")

    args = parser.parse_args()

    captioner = ImageCaptioner(args.size, args.checkpoint_name, args.res_path, args.temperature)
    captioner.set_image_path(args.img_path)
    caption = captioner.generate_caption()
    print(caption)

