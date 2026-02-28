[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)


# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)

Bộ công cụ Python để tạo chú thích ngôn ngữ tự nhiên cho hình ảnh và video bằng cách kết hợp embeddings thị giác của OpenAI CLIP với mô hình ngôn ngữ kiểu GPT.

## 🧭 Snapshot

| Kích thước | Chi tiết |
|---|---|
| Phạm vi nhiệm vụ | Chú thích ảnh và video |
| Kết quả chính | Phụ đề SRT, transcript JSON, ảnh có chú thích |
| Script chính | `i2c.py`, `v2c.py`, `image2caption.py` |
| Đường dẫn legacy | `video2caption.py` và các phiên bản phiên bản hoá (giữ lại cho mục đích tham khảo) |
| Luồng dữ liệu dataset | `data/raw/results.csv` + `data/raw/flickr30k_images/` |

## ✨ Tổng quan

Repository này cung cấp:

- Script suy luận cho việc tạo caption ảnh và phụ đề video.
- Pipeline huấn luyện học ánh xạ từ embeddings thị giác CLIP sang token embeddings của GPT-2.
- Tiện ích tạo dataset theo kiểu Flickr30k.
- Tự động tải checkpoint của các kích thước model được hỗ trợ khi thiếu trọng số.
- Các phiên bản README đa ngôn ngữ trong `i18n/` (xem thanh ngôn ngữ ở trên).

Triển khai hiện tại bao gồm cả script mới và script legacy. Một số file legacy được giữ lại để tham khảo và được mô tả bên dưới.

## 🚀 Tính năng

- Chú thích ảnh đơn qua `image2caption.py`.
- Chú thích video (lấy mẫu frame đồng đều) qua `v2c.py` hoặc `video2caption.py`.
- Tùy chỉnh tuỳ chọn chạy:
  - Số lượng frame.
  - Kích thước model.
  - Nhiệt độ sampling.
  - Tên checkpoint.
- Suy luận video song song/multi-thread để nhanh hơn.
- Tệp đầu ra:
  - Tệp phụ đề SRT (`.srt`).
  - Transcript JSON (`.json`) trong `v2c.py`.
- Điểm vào huấn luyện và đánh giá cho thí nghiệm ánh xạ CLIP+GPT2.

### Tóm tắt nhanh

| Mục | Script chính | Ghi chú |
|---|---|---|
| Chú thích ảnh | `image2caption.py`, `i2c.py`, `predict.py` | CLI + lớp dùng lại được |
| Chú thích video | `v2c.py` | Đường dẫn được duy trì, khuyến nghị |
| Dòng video legacy | `video2caption.py`, `video2caption_v1.1.py` | Chứa giả định phụ thuộc máy |
| Tạo dataset | `dataset_generation.py` | Tạo ra `data/processed/dataset.pkl` |
| Huấn luyện / đánh giá | `training.py`, `evaluate.py` | Dùng ánh xạ CLIP+GPT2 |

## 🧱 Kiến trúc (Mức tổng quan)

Mô hình lõi trong `model/model.py` gồm ba phần:

1. `ImageEncoder`: trích xuất embedding hình ảnh CLIP.
2. `Mapping`: biến đổi embedding CLIP thành một chuỗi embedding tiền tố của GPT.
3. `TextDecoder`: phần GPT-2 tự hồi quy sinh token caption.

Huấn luyện (`Net.train_forward`) dùng embeddings hình ảnh CLIP đã được tính trước cộng với caption đã tokenize.
Suy luận (`Net.forward`) dùng ảnh PIL và giải mã token cho đến khi gặp EOS hoặc `max_len`.

### Luồng dữ liệu

1. Chuẩn bị dataset: `dataset_generation.py` đọc `data/raw/results.csv` và ảnh trong `data/raw/flickr30k_images/`, ghi `data/processed/dataset.pkl`.
2. Huấn luyện: `training.py` nạp tuple pickle `(image_name, image_embedding, caption)` rồi huấn luyện các lớp mapper/decoder.
3. Đánh giá: `evaluate.py` render caption sinh ra trên ảnh test đã tách ra.
4. Cung cấp suy luận:
   - ảnh: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (khuyến nghị), `video2caption.py` (legacy).

## 🗂️ Cấu trúc dự án

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI caption ảnh đơn
├── predict.py                     # CLI caption ảnh đơn thay thế
├── i2c.py                         # Lớp ImageCaptioner dùng lại được + CLI
├── v2c.py                         # Video -> SRT + JSON (caption frame theo đa luồng)
├── video2caption.py               # Triển khai thay thế video -> SRT (ràng buộc legacy)
├── video2caption_v1.1.py          # Biến thể cũ hơn
├── video2caption_v1.0_not_work.py # File legacy được ghi rõ là không hoạt động
├── training.py                    # Điểm vào huấn luyện mô hình
├── evaluate.py                    # Đánh giá tập test và đầu ra đã render
├── dataset_generation.py          # Tạo data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Dataset + DataLoader helpers
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # Lớp tiện ích train/validation/test
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Mặc định ConfigS / ConfigL
│   ├── downloads.py               # Trình tải checkpoint Google Drive
│   └── lr_warmup.py               # Lịch trình LR warmup
├── i18n/                          # Các phiên bản README đa ngôn ngữ
└── .auto-readme-work/             # Tài sản/artefact của pipeline auto-readme
```

## 📋 Yêu cầu tiên quyết

- Python `3.10+` được khuyến nghị.
- GPU hỗ trợ CUDA không bắt buộc nhưng được khuyến nghị mạnh cho huấn luyện và suy luận model lớn.
- `ffmpeg` không bắt buộc trực tiếp bởi các script hiện tại (OpenCV được dùng để trích frame).
- Cần kết nối internet lần đầu để tải model/checkpoint từ Hugging Face / Google Drive.

Hiện chưa có lockfile (`requirements.txt` / `pyproject.toml` vắng mặt), nên dependency được suy ra từ các `import`.

## 🛠️ Cài đặt

### Thiết lập chuẩn theo cấu trúc repository hiện tại

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers pillow matplotlib numpy tqdm opencv-python pandas wandb gdown
```

### Đoạn cài đặt từ README gốc (được giữ nguyên)

README trước đó kết thúc giữa chừng trong một block. Các lệnh gốc giữ nguyên như nội dung lịch sử nguồn sau đây:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Lưu ý: snapshot repository hiện tại đặt script ở root repo, không nằm trong `src/`.

## ▶️ Bắt đầu nhanh

| Mục tiêu | Lệnh |
|---|---|
| Chú thích ảnh | `python image2caption.py -I /path/to/image.jpg -S L -C model.pt` |
| Chú thích video | `python v2c.py -V /path/to/video.mp4 -N 10` |
| Tạo dataset | `python dataset_generation.py` |

### Chú thích ảnh (chạy nhanh)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Chú thích video (đường dẫn khuyến nghị)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Cách sử dụng

### 1. Chú thích ảnh (`image2caption.py`)

```bash
python image2caption.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

Tham số:

- `-I, --img-path`: đường dẫn ảnh đầu vào.
- `-S, --size`: kích thước model (`S` hoặc `L`).
- `-C, --checkpoint-name`: tên checkpoint trong `weights/{small|large}`.
- `-R, --res-path`: thư mục đầu ra cho ảnh đã render caption.
- `-T, --temperature`: nhiệt độ sampling.

### 2. CLI ảnh thay thế (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` về chức năng tương tự `image2caption.py`; định dạng văn bản đầu ra có chênh lệch nhẹ.

### 3. API lớp chú thích ảnh (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Hoặc import trong script riêng của bạn:

```python
from i2c import ImageCaptioner

captioner = ImageCaptioner(model_size="L", checkpoint_name="model.pt")
captioner.set_image_path("/path/to/image.jpg")
caption = captioner.generate_caption(save_image=True)
print(caption)
```

### 4. Video thành phụ đề + JSON (`v2c.py`)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

Đầu ra nằm cạnh video đầu vào:

- `<video_basename>_caption.srt`
- `<video_basename>_caption.json`
- `<video_basename>_captioning_frames/`

### 5. Pipeline video thay thế (`video2caption.py`)

```bash
python video2caption.py -V /path/to/video.mp4 -N 10
```

Quan trọng: script hiện tại vẫn chứa các đường dẫn hardcoded phụ thuộc máy:

- Python path mặc định: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Đường dẫn script caption: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Hãy dùng `v2c.py` trừ khi bạn cố tình duy trì các đường dẫn này.

### 6. Biến thể legacy (`video2caption_v1.1.py`)

Script này được giữ lại để tham chiếu lịch sử. Với dùng thực tế, ưu tiên `v2c.py`.

### 7. Tạo dataset

```bash
python dataset_generation.py
```

Đầu vào thô dự kiến:

- `data/raw/results.csv` (bảng caption phân tách bằng dấu `|`).
- `data/raw/flickr30k_images/` (các file ảnh được CSV tham chiếu).

Đầu ra:

- `data/processed/dataset.pkl`

### 8. Huấn luyện

```bash
python training.py -S L -C model.pt
```

Huấn luyện mặc định dùng logging của Weights & Biases (`wandb`).

### 9. Đánh giá

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Đánh giá render caption dự đoán lên ảnh test và lưu tại:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Cấu hình

Cấu hình mô hình được định nghĩa trong `utils/config.py`:

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Giá trị mặc định từ các lớp cấu hình:

| Trường | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

ID tự động tải checkpoint nằm trong `utils/downloads.py`:

| Kích thước | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Tệp đầu ra

### Suy luận ảnh

- Ảnh có caption chồng/nội dung đè lên được lưu tại `--res-path`.
- Mẫu tên file: `<input_stem>-R<SIZE>.jpg`.

### Suy luận video (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Frame ảnh: `<video_stem>_captioning_frames/`

Ví dụ một phần tử JSON:

```json
{
  "start": "00:00:03,200",
  "end": "00:00:03,700",
  "lang": "en",
  "text": "A dog running through a field."
}
```

## 🧪 Ví dụ

### Ví dụ nhanh cho chú thích ảnh

```bash
python image2caption.py -I ./examples/dog.jpg -S S -C model.pt
```

Hành vi dự kiến:

- Nếu thiếu `weights/small/model.pt`, file sẽ được tải về.
- Mặc định một ảnh có caption được ghi vào `./data/result/prediction`.
- Văn bản caption được in ra stdout.

### Ví dụ nhanh cho chú thích video

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Hành vi dự kiến:

- 8 frame được lấy mẫu đồng đều để tạo caption.
- Tệp `.srt` và `.json` được tạo bên cạnh video đầu vào.

### Chuỗi huấn luyện/đánh giá end-to-end

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Ghi chú phát triển

- Có phần chồng lặp legacy giữa `v2c.py`, `video2caption.py`, và `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` được giữ lại có chủ đích như mã legacy không hoạt động.
- `training.py` hiện chọn `ConfigL()` qua `config = ConfigL() if args.size.upper() else ConfigS()`, luôn giải quyết về `ConfigL` cho mọi `--size` không rỗng.
- `model/trainer.py` dùng `self.dataset` trong `test_step`, trong khi hàm khởi tạo gán `self.test_dataset`; điều này có thể làm hỏng sampling trong các lần chạy huấn luyện nếu chưa chỉnh sửa.
- `video2caption_v1.1.py` tham chiếu `self.config.transform`, nhưng `ConfigS`/`ConfigL` không định nghĩa `transform`.
- Tạm thời chưa có CI/test suite trong snapshot repository hiện tại.
- Ghi chú i18n: các liên kết ngôn ngữ đã có ở đầu README; có thể bổ sung thêm bản dịch khác trong `i18n/`.
- Ghi chú trạng thái hiện tại: thanh ngôn ngữ liên kết đến `i18n/README.ru.md`, nhưng file này chưa có trong snapshot.

## 🩺 Xử lý sự cố

- `AssertionError: Image does not exist`
  - Kiểm tra `-I/--img-path` trỏ đến một file hợp lệ.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` nêu lỗi này khi `data/processed/dataset.pkl` chưa có; chạy `python dataset_generation.py` trước.
- `Path to the test image folder does not exist`
  - Kiểm tra `evaluate.py -I` trỏ đến folder hiện có.
- Chạy lần đầu chậm hoặc lỗi
  - Lần chạy đầu sẽ tải model từ Hugging Face và có thể tải checkpoint từ Google Drive.
- `video2caption.py` trả về caption rỗng
  - Kiểm tra đường dẫn script hardcode và Python executable, hoặc chuyển sang `v2c.py`.
- `wandb` yêu cầu đăng nhập khi huấn luyện
  - Chạy `wandb login` hoặc tắt logging thủ công trong `training.py` nếu cần.

## 🛣️ Lộ trình

- Thêm lockfile dependency (`requirements.txt` hoặc `pyproject.toml`) để cài đặt tái lập.
- Hợp nhất các pipeline video trùng lặp thành một triển khai duy nhất được duy trì.
- Loại bỏ hardcoded machine paths khỏi các script legacy.
- Sửa các bug biên nổi tiếng trong `training.py` và `model/trainer.py`.
- Thêm tests tự động và CI.
- Bổ sung đầy đủ `i18n/` với các README đã dịch được tham chiếu trong thanh ngôn ngữ.

## 🤝 Đóng góp

Đóng góp rất được hoan nghênh. Quy trình gợi ý:

```bash
# 1) Fork và clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Tạo nhánh tính năng
git checkout -b feat/your-change

# 3) Thực hiện thay đổi và commit
git add .
git commit -m "feat: describe your change"

# 4) Push và mở PR
git push origin feat/your-change
```

Nếu bạn thay đổi hành vi của model, hãy kèm theo:

- Lệnh có thể tái lập.
- Ví dụ đầu ra trước/sau.
- Ghi chú về giả định checkpoint hoặc dataset.

## ❤️ Support

| Donate | PayPal | Stripe |
|---|---|---|
| [![Donate](https://img.shields.io/badge/Donate-LazyingArt-0EA5E9?style=for-the-badge&logo=ko-fi&logoColor=white)](https://chat.lazying.art/donate) | [![PayPal](https://img.shields.io/badge/PayPal-RongzhouChen-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen) | [![Stripe](https://img.shields.io/badge/Stripe-Donate-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 Giấy phép

Không có tệp license trong snapshot repository hiện tại.

Lưu ý giả định: cho đến khi thêm tệp `LICENSE`, điều khoản tái sử dụng/phân phối vẫn chưa được xác định.
