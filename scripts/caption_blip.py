import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# 이미지 캡션 생성 함수
def generate_caption(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# 전체 폴더 처리
def caption_folder(image_dir, output_json, prefix=None):
    results = {}
    for fname in tqdm(sorted(os.listdir(image_dir))):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            path = os.path.join(image_dir, fname)
            image = Image.open(path)
            caption = generate_caption(image)
            if prefix:
                caption = prefix + " " + caption
            results[fname] = caption

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved {len(results)} captions to {output_json}")

if __name__ == "__main__":
    caption_folder("train_data", "captions_blip1.json", prefix="A TOFU_CHARACTER")
