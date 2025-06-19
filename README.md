
# SDXL LoRA Train & Inference

이 프로젝트는 **Stable Diffusion XL (SDXL)**을 기반으로 한 **DreamBooth + LoRA 학습 파이프라인**입니다.  
나를 닮은 특정 캐릭터를 생성하는 방법 
(img2img, 입력 이미지를 받아 텍스트로 묘사를 생성하며, 해당 텍스트와 캐릭터 트리거 키워드를 통해 입력 이미지와 닮은 캐릭터 이미지 생성성)

## 환경 설치

### 가상 환경 생성 (venv/conda)
```
[venv]
python -m venv sdxl_lora
source sdxl_lora/bin/activate      # (Linux/macOS)
sdxl_lora\Scripts\activate         # (Windows)

[conda]
conda create -n sdxl_lora python=3.10 -y
conda activate sdxl_lora
```

### PyTorch (CUDA 12.1용) 및 라이브러리 설치
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 (Windows)
pip install -r requirements.txt
accelerate config  # 첫 실행 시만
```
* 테스트 해보지 않았으나, CUDA 12 torch에서 모두 적용 가능할 것으로 예상

## 패키지 구조
```
| 폴더         | 설명                               |
| ---------- | -------------------------------- |
| `assets/`  | 추론 과정에서 입력 이미지로 사용할 이미지들 저장장|
| `data/`    | 학습 이미지, 캡션이 담긴 `metadata.jsonl`을 포함하는 폴더 (ex. data/TOFU)|
| `models/`  | SDXL/VAE/BLIP 모델, 학습된 LoRA 가중치 저장 |
| `scripts/` | 학습/캡션/추론용 Python 스크립트들           |
| `outputs/` | 생성된 이미지 출력 경로                    |
```

### 학습/추론론 시 요구 GPU 메모리
9626MiB 

## 모델 체크포인트
```
|-models
|----lora
|----madebyollin
|----Salesforce
|----stabilityai
```
- assets/package_structure.png 을 참고하여, 모델 파라미터를 다운로드

## 데이터 구축/모델학습/테스트 순서
- data/ 폴더(ex. data/TOFU)에 이미지 업로드 (이미지가 적을 경우 ChatGPT로 생성)
- scripts/caption_blip.py로 캡션 생성 및 외부 LLM (ChatGPT)로 캡션 전처리 (csv 파일 생성)
- scripts/caption_format_utils.py로 캡션 포맷 변경 (csv -> jsonl)
- scripts/train_text2img_lora_sdxl.py로 LoRA 학습 
- scripts/infer_img2img.py로 학습된 LoRA 테스트



### ChatGPT 기반 두부분식 캐릭터(학습용) 생성 과정
```
(두부분식 캐릭터 이미지 삽입과 함께) "1장 정도 두부 분식 스타일 캐릭터 이미지 생성해 줄 수 있어?"
(다양성 추가 요청 반복) "조금 더 이목구비나 변형 주면서, 해당 캐릭터의 친구들을 몇 개 만들어줄 수 있어?"
(다양성 추가 요청 반복) "좀 더 다양한 특색 있는 친구로 다양하게 만들어줘"
(실제 인물 기반 캐릭터화 요청) "이 친구 사진을 보고, 너가 여태까지 만든 두부 캐릭터처럼 만들어줘"
```
![alt text](assets/input_imgs.png)

## 두부분식 데이터 생성 및 캡셔닝 과정

### 1. 이미지 캡셔닝 (BLIP 사용)

* BLIP1 모델을 사용해 각 이미지(image\_1.png \~ image\_24.png)에 대한 초기 캡션을 자동 생성함.
* 예: `"image_1.png": "A TOFU_CHARACTER a purple background with a white square character"`

### 2. ChatGPT로 캡션 자연어 개선

* BLIP에서 생성한 어색한 표현(예: "a cartoon character with a mustache and mustache")이나 중복된 구조를 개선함.
* 불필요한 반복을 제거하고 "TOFU\_CHARACTER"를 의미 있는 이름인 "HANA\_TOFU"로 변경.

### 3. 파일명 정렬 및 문장 정제

* image\_1.png \~ image\_24.png까지 번호 순서대로 정렬하여 관리.
* 표현을 보다 자연스럽고 묘사 중심으로 변경 (예: "smiling while standing", "reading a book", "making an OK gesture").
* datasets/diffusers 버전에 따라서 jsonl의 필드명을 image가 아닌 file_name으로 사용해야 할 수도 있음 (Windows에서는 image가 아닌 file_name 사용)

### 4. 사용자 제공 설명 추가 반영

* 사용자가 직접 제공한 이미지 설명(예: "image\_6은 왼쪽 눈을 찡긋하면서 오른손을 들고 있는 두부야")을 기반으로 → 행동, 표정, 포즈 등을 구체적으로 반영한 문장으로 캡션 보강.
* 예시 변경 전: `"image_6.png": "A HANA_TOFU with a purple background"`
* → 변경 후: `"image_6.png": "A HANA_TOFU with a purple background, winking with its left eye and raising its right hand."`

### 5. 캡션 JSON 저장 (LoRA 학습용)

* 최종 `{filename: caption}` 형태로 저장된 JSON은 LoRA fine-tuning에서 `--caption_file`로 바로 사용 가능.
* LoRA, DreamBooth, 또는 `datasets.load_dataset()` 등에서 사용할 수 있도록 하기 위해서는 Hugging Face의 공식 포맷(`.jsonl`, `.csv` 등)을 따라야 함.

#### 예시: `.jsonl` 포맷 (각 줄이 JSON 객체)

```json
{"file_name": "image_1.png", "text": "A HANA_TOFU with a purple background, smiling..."}
{"file_name": "image_2.png", "text": "A HANA_TOFU with a purple background, wearing glasses."}
```

 또는 `.csv` 포맷으로 변환:

```csv
file_name,text
image_1.png,A HANA_TOFU with a purple background, smiling...
image_2.png,A HANA_TOFU with a purple background, wearing glasses.
```
### 학습 방법
- args는 적절하게 변경 가능
```
accelerate launch scripts/train_t2i_lora_sdxl.py --pretrained_model_name_or_path="models/stabilityai/stable-diffusion-xl-base-1.0" --pretrained_vae_model_name_or_path="models/madebyollin/sdxl-vae-fp16-fix" --output_dir="HANA_TOFU_LoRA" --caption_column="text" --dataset_name="imagefolder" --train_data_dir="data/TOFU" --mixed_precision="fp16" --validation_prompt="A HANA_TOFU of a man wearing a virtual reality headset" --resolution=1024 --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing --learning_rate=1e-4 --snr_gamma=5.0 --lr_scheduler="constant" --lr_warmup_steps=0 --use_8bit_adam --max_train_steps=500 --checkpointing_steps=717 --seed="0"
```
### 추론 방법
- args는 적절하게 변경 가능
```
python -m scripts.infer_img2img
```

![alt text](assets/output_img.png)
---

## 라이선스
- HuggingFace diffusers 기반
- StabilityAI SDXL 모델 사용 

