import subprocess
import cv2
import os
import concurrent.futures

class VideoToSRTConverter:
    def __init__(self, video_path, num_frames=10, conda_env_path="/home/lachlan/miniconda3/envs/caption/bin/python"):
        self.video_path = video_path
        self.num_frames = num_frames
        self.conda_env_path = conda_env_path
        # Modify output directory to include a specific folder for frames
        self.frames_dir = os.path.splitext(video_path)[0] + "_captioning_frames"
        self.output_srt = os.path.splitext(video_path)[0] + "_caption.srt"
        os.makedirs(self.frames_dir, exist_ok=True)  # Ensure the directory is created

    def extract_frame_at_timestamp(self, timestamp, frame_id):
        frame_path = os.path.join(self.frames_dir, f"frame_{frame_id:04d}.jpg")
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * timestamp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frame_path, frame)
        cap.release()
        return frame_path

    def caption_frame(self, frame_path):
        script_path = '/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py'
        cmd = [self.conda_env_path, script_path, '-I', frame_path, '-S', 'L', '-C', 'model.pt']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip() if result.stdout else "No caption generated"

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

    def convert(self):
        fps = cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FPS)
        total_frames = int(cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        timestamps = [i * (duration / self.num_frames) for i in range(self.num_frames)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            frames = [executor.submit(self.extract_frame_at_timestamp, timestamp, i)
                      for i, timestamp in enumerate(timestamps)]

        srt_entries = [{
            'index': i + 1,
            'start': self.format_time(i * (duration / self.num_frames)),
            'end': self.format_time((i + 1) * (duration / self.num_frames)),
            'text': self.caption_frame(frames[i].result())
        } for i in range(self.num_frames)]

        self.save_srt_file(srt_entries)
        print(f"SRT file saved to {self.output_srt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video to Captions as SRT")
    parser.add_argument("-V", "--video-path", type=str, required=True, help="Path to the video file")
    parser.add_argument("-N", "--num-frames", type=int, default=10, help="Number of frames to caption")
    args = parser.parse_args()

    converter = VideoToSRTConverter(args.video_path, num_frames=args.num_frames)
    converter.convert()

