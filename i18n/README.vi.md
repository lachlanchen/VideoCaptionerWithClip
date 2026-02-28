[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Clip-GPT-Captioning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/README-Expanded-success)
![Repo Layout](https://img.shields.io/badge/Layout-Root%20Scripts-informational)
![Legacy Scripts](https://img.shields.io/badge/Legacy%20Scripts-Present-orange)
![i18n](https://img.shields.io/badge/i18n-Enabled-brightgreen)
![Maintained Path](https://img.shields.io/badge/Video-v2c.py-2ea44f)

Bộ công cụ Python để tạo chú thích ngôn ngữ tự nhiên cho hình ảnh và video bằng cách kết hợp embedding thị giác OpenAI CLIP với mô hình ngôn ngữ kiểu GPT.

## ✨ Tổng quan

Kho lưu trữ này cung cấp:

- Script suy luận cho chú thích ảnh và tạo phụ đề video.
- Pipeline huấn luyện học ánh xạ từ CLIP visual embeddings sang GPT-2 token embeddings.
- Tiện ích tạo dataset theo kiểu Flickr30k.
- Tự động tải checkpoint cho các kích thước model được hỗ trợ khi thiếu weights.
- Các biến thể README đa ngôn ngữ trong `i18n/` (xem thanh ngôn ngữ ở trên).

Bản triển khai hiện tại bao gồm cả script mới và script legacy. Một số file legacy được giữ lại để tham khảo và được mô tả bên dưới.

## 🚀 Tính năng

- Tạo caption cho một ảnh qua `image2caption.py`.
- Tạo caption cho video (lấy mẫu frame đồng đều) qua `v2c.py` hoặc `video2caption.py`.
- Tùy chỉnh các tùy chọn runtime:
  - Số lượng frame.
  - Kích thước model.
  - Nhiệt độ lấy mẫu.
  - Tên checkpoint.
- Caption đa tiến trình/đa luồng để suy luận video nhanh hơn.
- Tệp đầu ra:
  - Tệp phụ đề SRT (`.srt`).
  - Transcript JSON (`.json`) trong `v2c.py`.
- Điểm vào huấn luyện và đánh giá cho thí nghiệm ánh xạ CLIP+GPT2.

### Tóm tắt nhanh

| Khu vực | Script chính | Ghi chú |
|---|---|---|
| Chú thích ảnh | `image2caption.py`, `i2c.py`, `predict.py` | CLI + lớp tái sử dụng |
| Chú thích video | `v2c.py` | Đường dẫn được duy trì, khuyến nghị dùng |
| Luồng video legacy | `video2caption.py`, `video2caption_v1.1.py` | Chứa các giả định phụ thuộc máy cụ thể |
| Tạo dataset | `dataset_generation.py` | Tạo `data/processed/dataset.pkl` |
| Huấn luyện / đánh giá | `training.py`, `evaluate.py` | Dùng ánh xạ CLIP+GPT2 |

## 🧱 Kiến trúc (Mức cao)

Mô hình cốt lõi trong `model/model.py` có ba phần:

1. `ImageEncoder`: trích xuất CLIP image embedding.
2. `Mapping`: chiếu CLIP embedding thành chuỗi GPT prefix embedding.
3. `TextDecoder`: đầu ra mô hình ngôn ngữ GPT-2 để tự hồi quy sinh token caption.

Huấn luyện (`Net.train_forward`) dùng CLIP image embeddings đã tính trước + caption đã token hóa.
Suy luận (`Net.forward`) dùng ảnh PIL và giải mã token đến EOS hoặc `max_len`.

### Luồng dữ liệu

1. Chuẩn bị dataset: `dataset_generation.py` đọc `data/raw/results.csv` và ảnh trong `data/raw/flickr30k_images/`, ghi `data/processed/dataset.pkl`.
2. Huấn luyện: `training.py` nạp tuple pickle `(image_name, image_embedding, caption)` và huấn luyện các lớp mapper/decoder.
3. Đánh giá: `evaluate.py` render caption được tạo lên các ảnh test hold-out.
4. Phục vụ suy luận:
   - ảnh: `image2caption.py` / `predict.py` / `i2c.py`.
   - video: `v2c.py` (khuyến nghị), `video2caption.py` (legacy).

## 🗂️ Cấu trúc dự án

```text
VideoCaptionerWithClip/
├── README.md
├── image2caption.py               # CLI caption ảnh đơn
├── predict.py                     # CLI caption ảnh đơn thay thế
├── i2c.py                         # Lớp ImageCaptioner tái sử dụng + CLI
├── v2c.py                         # Video -> SRT + JSON (caption frame đa luồng)
├── video2caption.py               # Cài đặt Video -> SRT thay thế (ràng buộc legacy)
├── video2caption_v1.1.py          # Biến thể cũ hơn
├── video2caption_v1.0_not_work.py # File legacy được đánh dấu rõ là không hoạt động
├── training.py                    # Điểm vào huấn luyện model
├── evaluate.py                    # Đánh giá tập test và đầu ra đã render
├── dataset_generation.py          # Tạo data/processed/dataset.pkl
├── data/
│   ├── __init__.py
│   └── dataset.py                 # Bộ trợ giúp Dataset + DataLoader
├── model/
│   ├── __init__.py
│   ├── model.py                   # CLIP encoder + mapping + GPT2 decoder
│   └── trainer.py                 # Lớp tiện ích huấn luyện/xác thực/kiểm thử
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Mặc định ConfigS / ConfigL
│   ├── downloads.py               # Trình tải checkpoint từ Google Drive
│   └── lr_warmup.py               # Lịch LR warmup
├── i18n/                          # Các biến thể README đa ngôn ngữ
└── .auto-readme-work/             # Tạo tác pipeline Auto-README
```

## 📋 Điều kiện tiên quyết

- Khuyến nghị Python `3.10+`.
- GPU hỗ trợ CUDA là tùy chọn nhưng rất nên có cho huấn luyện và suy luận model lớn.
- `ffmpeg` không bắt buộc trực tiếp với script hiện tại (dùng OpenCV để trích frame).
- Cần internet ở lần chạy đầu để tải model/checkpoint từ Hugging Face / Google Drive.

Hiện chưa có lockfile (`requirements.txt` / `pyproject.toml` thiếu), nên dependency được suy ra từ các import.

## 🛠️ Cài đặt

### Thiết lập chuẩn từ cấu trúc repo hiện tại

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

README trước đó kết thúc giữa chừng trong một block. Các lệnh gốc được giữ nguyên bên dưới như nội dung lịch sử nguồn chuẩn:

```bash
git clone git@github.com:lachlanchen/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip/src
```

Lưu ý: bản snapshot repo hiện tại đặt script ở root repo, không nằm trong `src/`.

## ▶️ Bắt đầu nhanh

### Chú thích ảnh (chạy nhanh)

```bash
python image2caption.py -I /path/to/image.jpg -S L -C model.pt
```

### Chú thích video (đường dẫn khuyến nghị)

```bash
python v2c.py -V /path/to/video.mp4 -N 10
```

## 🎯 Cách dùng

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
- `-C, --checkpoint-name`: tên file checkpoint trong `weights/{small|large}`.
- `-R, --res-path`: thư mục đầu ra cho ảnh đã render caption.
- `-T, --temperature`: nhiệt độ lấy mẫu.

### 2. CLI ảnh thay thế (`predict.py`)

```bash
python predict.py \
  -I /path/to/image.jpg \
  -S L \
  -C model.pt \
  -R ./data/result/prediction \
  -T 1.0
```

`predict.py` tương tự về chức năng với `image2caption.py`; định dạng văn bản đầu ra khác nhẹ.

### 3. API lớp chú thích ảnh (`i2c.py`)

```bash
python i2c.py -I /path/to/image.jpg -S L -C model.pt -R ./data/result/prediction -T 1.0
```

Hoặc import trong script của bạn:

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

Quan trọng: script này hiện chứa các đường dẫn hardcode phụ thuộc máy:

- Python path mặc định: `/home/lachlan/miniconda3/envs/caption/bin/python`
- Đường dẫn script caption: `/home/lachlan/Projects/image_captioning/clip-gpt-captioning/src/image2caption.py`

Dùng `v2c.py` trừ khi bạn chủ đích duy trì các đường dẫn này.

### 6. Biến thể legacy (`video2caption_v1.1.py`)

Script này được giữ lại để tham chiếu lịch sử. Với nhu cầu dùng thực tế, hãy ưu tiên `v2c.py`.

### 7. Tạo dataset

```bash
python dataset_generation.py
```

Đầu vào raw dự kiến:

- `data/raw/results.csv` (bảng caption phân tách bằng pipe).
- `data/raw/flickr30k_images/` (các tệp ảnh được CSV tham chiếu).

Đầu ra:

- `data/processed/dataset.pkl`

### 8. Huấn luyện

```bash
python training.py -S L -C model.pt
```

Huấn luyện mặc định dùng logging Weights & Biases (`wandb`).

### 9. Đánh giá

```bash
python evaluate.py \
  -I ./data/raw/flickr30k_images \
  -R ./data/result/eval \
  -S L \
  -C model.pt \
  -T 1.0
```

Đánh giá sẽ render caption dự đoán lên ảnh test và lưu tại:

- `<res-path>/<checkpoint_name_without_ext>_<SIZE>/`

## ⚙️ Cấu hình

Cấu hình model được định nghĩa trong `utils/config.py`:

| Config | CLIP backbone | GPT model | Weights dir |
|---|---|---|---|
| `ConfigS` | `openai/clip-vit-base-patch32` | `gpt2` | `weights/small` |
| `ConfigL` | `openai/clip-vit-large-patch14` | `gpt2-medium` | `weights/large` |

Các giá trị mặc định chính từ lớp config:

| Field | `ConfigS` | `ConfigL` |
|---|---:|---:|
| `epochs` | 150 | 120 |
| `lr` | 3e-3 | 5e-3 |
| `batch_size_exp` | 6 | 5 |
| `ep_len` | 4 | 4 |
| `max_len` | 40 | 40 |

ID tự động tải checkpoint nằm trong `utils/downloads.py`:

| Size | Google Drive ID |
|---|---|
| `L` | `1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG` |
| `S` | `1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF` |

## 📦 Tệp đầu ra

### Suy luận ảnh

- Ảnh được lưu với tiêu đề đã chèn/tạo tại `--res-path`.
- Mẫu tên tệp: `<input_stem>-R<SIZE>.jpg`.

### Suy luận video (`v2c.py`)

- SRT: `<video_stem>_caption.srt`
- JSON: `<video_stem>_caption.json`
- Ảnh frame: `<video_stem>_captioning_frames/`

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

- Nếu thiếu `weights/small/model.pt`, tệp sẽ được tải về.
- Ảnh có caption mặc định được ghi vào `./data/result/prediction`.
- Văn bản caption được in ra stdout.

### Ví dụ nhanh cho chú thích video

```bash
python v2c.py -V ./examples/demo.mp4 -N 8
```

Hành vi dự kiến:

- 8 frame lấy mẫu đồng đều sẽ được tạo caption.
- Tệp `.srt` và `.json` được tạo cạnh video đầu vào.

### Chuỗi huấn luyện/đánh giá end-to-end

```bash
python dataset_generation.py
python training.py -S L -C model.pt
python evaluate.py -I ./data/raw/flickr30k_images -R ./data/result/eval -S L -C model.pt -T 1.0
```

## 🧭 Ghi chú phát triển

- Có sự chồng lặp legacy giữa `v2c.py`, `video2caption.py` và `video2caption_v1.*`.
- `video2caption_v1.0_not_work.py` được giữ lại có chủ đích như mã legacy không hoạt động.
- `training.py` hiện chọn `ConfigL()` qua `config = ConfigL() if args.size.upper() else ConfigS()`, luôn trả về `ConfigL` với mọi giá trị `--size` không rỗng.
- `model/trainer.py` dùng `self.dataset` trong `test_step`, trong khi hàm khởi tạo gán `self.test_dataset`; điều này có thể làm hỏng lấy mẫu trong các lần chạy huấn luyện nếu chưa chỉnh sửa.
- `video2caption_v1.1.py` tham chiếu `self.config.transform`, nhưng `ConfigS`/`ConfigL` không định nghĩa `transform`.
- Không có CI/test suite được định nghĩa trong snapshot repo hiện tại.
- Ghi chú i18n: liên kết ngôn ngữ có ở đầu README này; các tệp dịch có thể được thêm dưới `i18n/`.
- Ghi chú trạng thái hiện tại: thanh ngôn ngữ liên kết tới `i18n/README.ru.md`, nhưng tệp đó không có trong snapshot này.

## 🩺 Khắc phục sự cố

- `AssertionError: Image does not exist`
  - Xác nhận `-I/--img-path` trỏ tới tệp hợp lệ.
- `Dataset file not found. Downloading...`
  - `MiniFlickrDataset` sẽ báo lỗi này khi thiếu `data/processed/dataset.pkl`; hãy chạy `python dataset_generation.py` trước.
- `Path to the test image folder does not exist`
  - Xác nhận `evaluate.py -I` trỏ tới thư mục tồn tại.
- Lần chạy đầu chậm hoặc thất bại
  - Lần chạy đầu sẽ tải model từ Hugging Face và có thể tải checkpoint từ Google Drive.
- `video2caption.py` trả về caption rỗng
  - Kiểm tra đường dẫn script hardcode và đường dẫn trình thông dịch Python, hoặc chuyển sang `v2c.py`.
- `wandb` yêu cầu đăng nhập trong lúc huấn luyện
  - Chạy `wandb login` hoặc tắt logging thủ công trong `training.py` nếu cần.

## 🛣️ Lộ trình

- Thêm lockfile dependency (`requirements.txt` hoặc `pyproject.toml`) để cài đặt có thể tái lập.
- Hợp nhất các pipeline video trùng lặp thành một bản triển khai được duy trì.
- Loại bỏ đường dẫn máy hardcode khỏi script legacy.
- Sửa các lỗi biên đã biết của huấn luyện/đánh giá trong `training.py` và `model/trainer.py`.
- Thêm kiểm thử tự động và CI.
- Hoàn thiện `i18n/` với các README dịch được tham chiếu trong thanh ngôn ngữ.

## 🤝 Đóng góp

Hoan nghênh đóng góp. Quy trình gợi ý:

```bash
# 1) Fork and clone
git clone git@github.com:<your-user>/VideoCaptionerWithClip.git
cd VideoCaptionerWithClip

# 2) Create a feature branch
git checkout -b feat/your-change

# 3) Make changes and commit
git add .
git commit -m "feat: describe your change"

# 4) Push and open a PR
git push origin feat/your-change
```

Nếu bạn thay đổi hành vi model, hãy kèm theo:

- (Các) lệnh có thể tái lập.
- Mẫu đầu ra trước/sau.
- Ghi chú về các giả định checkpoint hoặc dataset.

## 🙌 Hỗ trợ

Không có cấu hình donation/sponsorship rõ ràng trong snapshot repo hiện tại.

Nếu liên kết tài trợ được thêm sau này, chúng nên được giữ trong phần này.

## 📄 Giấy phép

Không có tệp license trong snapshot repo hiện tại.

Ghi chú giả định: cho đến khi có tệp `LICENSE`, điều khoản tái sử dụng/phân phối vẫn chưa được xác định.
