# scripts/inference_img2img.py

import os
import re
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import DiffusionPipeline, AutoencoderKL, DDIMScheduler


def clean_caption(caption: str) -> str:
    caption = caption.lower().strip().rstrip(".")
    caption = re.sub(r"\b(a|an|the|girl|boy|woman|man|person|child)\b", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def generate_caption_fn(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    def caption(image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(output_ids[0], skip_special_tokens=True).strip()

    return caption


def generate_images_from_folder_img2img(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # VAE + Pipe
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(
        args.base_model,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora_path, weight_name=args.lora_weight_name)
    pipe.fuse_lora()

    # Caption 모델 (옵션)
    if args.use_caption:
        generate_caption = generate_caption_fn(device)

    transform = T.Compose([
        T.Resize((args.resolution, args.resolution)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    for fname in tqdm(sorted(os.listdir(args.image_dir))):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        input_path = os.path.join(args.image_dir, fname)
        image = Image.open(input_path).convert("RGB")

        # Caption
        if args.use_caption:
            caption = generate_caption(image)
            cleaned = clean_caption(caption)
            full_prompt = f"{args.prompt} {cleaned}"
        else:
            full_prompt = args.prompt

        print(f"📝 Prompt: {full_prompt}")

        # Latent 생성
        image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        t = torch.tensor([int(pipe.scheduler.config.num_train_timesteps * args.strength)], device=device, dtype=torch.long)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

        # 이미지 생성
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            image_output = pipe(
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                latents=noisy_latents
            ).images[0]

        out_path = os.path.join(args.output_dir, f"tofu_{os.path.splitext(fname)[0]}.png")
        image_output.save(out_path)

        if args.concat:
            output_w, output_h = image_output.size
            input_w, input_h = image.size
            ratio = min(output_w / input_w, output_h / input_h)
            resized_input = image.resize((int(input_w * ratio), int(input_h * ratio)), Image.LANCZOS)

            padded_input = Image.new("RGB", (output_w, output_h), (0, 0, 0))
            offset_x = (output_w - resized_input.width) // 2
            offset_y = (output_h - resized_input.height) // 2
            padded_input.paste(resized_input, (offset_x, offset_y))

            concatenated = Image.new("RGB", (output_w * 2, output_h))
            concatenated.paste(padded_input, (0, 0))
            concatenated.paste(image_output, (output_w, 0))

            concat_out_path = os.path.join(args.output_dir, f"concat_tofu_{os.path.splitext(fname)[0]}.png")
            concatenated.save(concat_out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Tofu-style face img2img generator (SDXL + LoRA)")
    parser.add_argument("--image_dir", type=str, default="assets/images")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--base_model", type=str, default="models/stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model", type=str, default="models/madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--lora_path", type=str, default="models/lora")
    parser.add_argument("--lora_weight_name", type=str, default="pytorch_lora_weights.safetensors")
    parser.add_argument("--prompt", type=str, default="A cute HANA_TOFU character with white square face, purple background, 2D cartoon vector style, kawaii")
    parser.add_argument("--negative_prompt", type=str, default="realistic, human, skin texture, watermark, glitch, extra limbs, nsfw, text, logo")
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--concat", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_images_from_folder_img2img(args)
