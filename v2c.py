import cv2
import os
import argparse
import json
import concurrent.futures
from i2c import ImageCaptioner  # Importing ImageCaptioner from i2c.py

class VideoToCaption:
    def __init__(self, video_path, num_frames=3, model_size='L', checkpoint_name='model.pt', temperature=1.0):
        self.video_path = video_path
        self.num_frames = num_frames
        self.model_size = model_size
        self.checkpoint_name = checkpoint_name
        self.temperature = temperature
        self.frames_dir = os.path.splitext(video_path)[0] + "_captioning_frames"
        self.output_srt = os.path.splitext(video_path)[0] + "_caption.srt"
        self.output_json = os.path.splitext(video_path)[0] + "_caption.json"
        os.makedirs(self.frames_dir, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = total_frames / fps
        timestamps = [i * (self.duration / self.num_frames) for i in range(self.num_frames)]
        frames = []
        for i, timestamp in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * timestamp))
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
        cap.release()
        return frames

    def caption_frame(self, frame_path):
        captioner = ImageCaptioner(model_size=self.model_size, checkpoint_name=self.checkpoint_name, res_path=self.frames_dir, temperature=self.temperature)
        captioner.set_image_path(frame_path)
        caption = captioner.generate_caption()

        del captioner
        return caption

    def convert(self):
        frames = self.extract_frames()
        srt_entries = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.caption_frame, frame): i for i, frame in enumerate(frames)}
            for future in concurrent.futures.as_completed(futures):
                frame_index = futures[future]
                caption = future.result()
                srt_entries.append({
                    'index': frame_index + 1,
                    'start': self.format_time(frame_index * (self.duration / self.num_frames)),
                    'end': self.format_time((frame_index + 0.5) * (self.duration / self.num_frames)),
                    'text': caption
                })

        srt_entries.sort(key=lambda x: x['index'])  # Ensure SRT entries are sorted
        self.save_srt_file(srt_entries)
        self.save_json_file(srt_entries)

    def format_time(self, seconds):
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), seconds % 60
        ms = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{int(s):02},{ms:03}"

    def save_srt_file(self, srt_entries):
        with open(self.output_srt, 'w') as file:
            for entry in srt_entries:
                file.write(f"{entry['index']}\n")
                file.write(f"{entry['start']} --> {entry['end']}\n")
                file.write(f"{entry['text']}\n\n")
        print(f"SRT file saved to {self.output_srt}")

    def save_json_file(self, srt_entries):
        json_data = [{
            "start": entry['start'],
            "end": entry['end'],
            "lang": "en",  # Assuming language is English; adjust as necessary
            "text": entry['text']
        } for entry in srt_entries]
        with open(self.output_json, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON file saved to {self.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to SRT and JSON captions")
    parser.add_argument("-V", "--video-path", type=str, required=True, help="Path to the video file")
    parser.add_argument("-N", "--num-frames", type=int, default=10, help="Number of frames to caption")
    args = parser.parse_args()

    converter = VideoToCaption(args.video_path, num_frames=args.num_frames)
    converter.convert()
