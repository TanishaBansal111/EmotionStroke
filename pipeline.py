# -*- coding: utf-8 -*-
"""gen-ai-project.ipynb

## Gen AI (UCS748) Project - EmotionStroke

### Tanisha &  Bhavneet Kaur
"""

template = (
    "an expressive artwork, {emotion} mood, {valence} valence, {arousal} arousal, "
    "{color} colors, {lighting} lighting, {composition}, in a painterly style, depicting {situation}\n"
)

defaults = [
    "# Defaults:",
    "# color=harmonious",
    "# lighting=soft",
    "# composition=balanced composition",
    "# situation=an evocative scene",
    ""
]

with open("prompts.txt", "w", encoding="utf-8") as f:
    f.write(template)
    f.write("\n".join(defaults))
print("prompts.txt created")

# data.py
import os
import io
import tarfile
import random
import subprocess
from typing import Tuple, Optional

import pandas as pd
from PIL import Image
from datasets import load_dataset
from pathlib import Path

HF_DATASET = "printblue/EmoArt-5k"
IMAGES_ARCHIVE_URL = "https://huggingface.co/datasets/printblue/EmoArt-5k/resolve/main/Images.tar.gz"
IMAGES_ARCHIVE_NAME = "Images.tar.gz"
IMAGES_DIR_NAME = "Images"

def _kaggle_default_root() -> str:
    root = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
    return root if os.path.exists(root) else os.getcwd()

def ensure_images_downloaded(img_root: Optional[str] = None) -> str:
    base = img_root or _kaggle_default_root()
    images_dir = os.path.join(base, IMAGES_DIR_NAME)
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        print(f"[INFO] Found Images directory at: {images_dir}")
        return base

    os.makedirs(base, exist_ok=True)
    archive_path = os.path.join(base, IMAGES_ARCHIVE_NAME)

    print(f"[INFO] Downloading {IMAGES_ARCHIVE_NAME} to {archive_path} ...")
    cmd = ["wget", "-O", archive_path, IMAGES_ARCHIVE_URL]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise RuntimeError(f"wget failed: {e}. Ensure internet is enabled and URL is reachable.")

    print(f"[INFO] Extracting {archive_path} ...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=base)
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}. Corrupted archive or insufficient space.")

    print(f"[INFO] Images extracted to: {images_dir}")
    return base

def _normalize_join(base: str, rel_path: str) -> str:
    rel_norm = (rel_path or "").replace("\\", "/")
    return os.path.join(base, rel_norm)

def _open_image(fp: str) -> Optional[Image.Image]:
    try:
        if os.path.exists(fp):
            return Image.open(fp).convert("RGB")
    except Exception:
        return None
    return None

def _parse_record(r: dict) -> Optional[dict]:
    img_rel = r.get("image_path", None)
    desc = r.get("description", {}) or {}
    if not isinstance(desc, dict):
        return None

    first = desc.get("first_section", {}) or {}
    second = desc.get("second_section", {}) or {}
    va = second.get("visual_attributes", {}) or {}
    third = desc.get("third_section", {}) or {}

    emo = third.get("dominant_emotion", None)
    val = third.get("emotional_valence", None)
    aro = third.get("emotional_arousal_level", None)

    color = va.get("color", "") or ""
    lighting = va.get("light_and_shadow", "") or ""
    composition = va.get("composition", "") or ""
    situation = first.get("description", "") or ""

    if any(v is None for v in [img_rel, emo, val, aro]):
        return None

    return {
        "image_rel": img_rel,
        "emotion": emo,
        "valence": val,
        "arousal": aro,
        "color": color,
        "lighting": lighting,
        "composition": composition,
        "situation": situation,
    }

def load_emoart(split_ratio: float = 0.9, max_samples: Optional[int] = None, seed: int = 42,
                img_root: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1) Ensure Images/ available
    base = ensure_images_downloaded(img_root)

    # 2) Load HF dataset metadata
    ds = load_dataset(HF_DATASET, split="train")
    if len(ds) == 0:
        raise RuntimeError("EmoArt-5k returned zero rows; check network access.")

    # 3) Parse rows
    records = []
    for r in ds:
        rec = _parse_record(r)
        if rec is None:
            continue
        fp = _normalize_join(base, rec["image_rel"])
        if not Path(fp).exists():
            continue
        records.append({
            "image_path": fp,
            "emotion": rec["emotion"],
            "valence": rec["valence"],
            "arousal": rec["arousal"],
            "color": rec["color"],
            "lighting": rec["lighting"],
            "composition": rec["composition"],
            "situation": rec["situation"],
        })

    if not records:
        raise RuntimeError("Parsed 0 usable rows; check Images/ and image paths.")

    # 4) Optional subsample
    if max_samples is not None:
        random.seed(seed)
        records = random.sample(records, min(max_samples, len(records)))

    df = pd.DataFrame(records)

    # 5) Train/val split
    cut = max(1, int(len(df) * split_ratio))
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

train_df, val_df = load_emoart(split_ratio=0.9, max_samples=2000, seed=42)

print("Train set size:", len(train_df))
print("Validation set size:", len(val_df))

train_df.head()

import os

root_dir = "/kaggle/working/Images"

file_count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    file_count += len(filenames)

print(f"Total number of files in '{root_dir}' (including subdirectories): {file_count}")

# generate train_emoart_paths.csv
from datasets import load_dataset
import os, pandas as pd

IMG_ROOT = "/kaggle/working"

ds = load_dataset("printblue/EmoArt-5k", split="train")
rows = []
for r in ds:
    rel = (r.get("image_path") or "").replace("\\", "/")
    fp = os.path.join(IMG_ROOT, rel) if rel else None

    desc = r.get("description") or {}
    first = desc.get("first_section", {}) if isinstance(desc, dict) else {}
    second = desc.get("second_section", {}) if isinstance(desc, dict) else {}
    va = second.get("visual_attributes", {}) if isinstance(second, dict) else {}
    third = desc.get("third_section", {}) if isinstance(desc, dict) else {}

    emo = third.get("dominant_emotion")
    val = third.get("emotional_valence")
    aro = third.get("emotional_arousal_level")
    color = va.get("color") or ""
    lighting = va.get("light_and_shadow") or ""
    comp = va.get("composition") or ""
    situation = first.get("description") or ""

    if fp and os.path.exists(fp) and emo and val and aro:
        rows.append({
            "image_path": fp,
            "emotion": emo,
            "valence": val,
            "arousal": aro,
            "color": color,
            "lighting": lighting,
            "composition": comp,
            "situation": situation,
        })

assert rows, "No rows found; ensure IMG_ROOT points to the directory that contains the Images/ folder."
pd.DataFrame(rows).to_csv("train_emoart_paths.csv", index=False)
print("Wrote train_emoart_paths.csv with", len(rows), "rows")

import pandas as pd
df = pd.read_csv("train_emoart_paths.csv")
df.head()

# pipeline
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

# ------------- prompt template loader -------------
def read_template(path="prompts.txt"):
    template = None
    defaults = {"color":"harmonious","lighting":"soft","composition":"balanced composition","situation":"an evocative scene"}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        for ln in lines:
            if all(k in ln for k in ["{emotion}", "{valence}", "{arousal}", "{situation}"]):
                template = ln
                break
    if not template:
        template = "an expressive artwork, {emotion} mood, {valence} valence, {arousal} arousal, {color} colors, {lighting} lighting, {composition}, in a painterly style, depicting {situation}"
    return template, defaults

def build_prompt(template, defaults, emotion, valence, arousal, color, lighting, composition, situation):
    return template.format(
        emotion=str(emotion).lower(),
        valence=str(valence).lower(),
        arousal=str(arousal).lower(),
        color=color or defaults["color"],
        lighting=lighting or defaults["lighting"],
        composition=composition or defaults["composition"],
        situation=situation or defaults["situation"],
    )  # [web:8]

# ------------- Lightweight image pre/post -------------
def pil_to_tensor(image: Image.Image, size: int = 384, center_crop=True):
    image = image.convert("RGB")
    if center_crop:
        min_side = min(image.size)
        left = (image.width - min_side) // 2
        top = (image.height - min_side) // 2
        image = image.crop((left, top, left + min_side, top + min_side))
    image = image.resize((size, size), Image.BICUBIC)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[None].transpose(0, 3, 1, 2)
    arr = 2.0 * arr - 1.0
    return torch.from_numpy(arr)

# ------------- Dataset -------------
class EmoArtSet(Dataset):
    def __init__(self, df, tokenizer, size=384, center_crop=True, template=None, defaults=None):
        self.df = df
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.template = template
        self.defaults = defaults

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img_path = r["image_path"]
        img = Image.open(img_path).convert("RGB")

        prompt = build_prompt(
            self.template, self.defaults,
            r["emotion"], r["valence"], r["arousal"],
            r.get("color",""), r.get("lighting",""), r.get("composition",""), r.get("situation","")
        )
        pixel_values = pil_to_tensor(img, self.size, self.center_crop)[0]
        tok = self.tokenizer(
            prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tok.input_ids[0],
            "attention_mask": tok.attention_mask[0],
        }

# ------------- LoRA utilities -------------

def add_lora(pipe, rank=16, text_enc=False):
    try:
        from peft import LoraConfig
    except:
        from diffusers.utils import LoraConfig

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(unet_lora_config)

    if text_enc:
        te_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        pipe.text_encoder.add_adapter(te_lora_config)

    return pipe

def iter_lora_params(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p


# ------------- Training -------------
def train(args):
    torch.manual_seed(args.seed)

    template, defaults = read_template(args.prompts)
    import pandas as pd
    assert os.path.exists(args.train_csv), f"Missing {args.train_csv}"
    df = pd.read_csv(args.train_csv)

    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    # Split
    cut = max(1, int(len(df) * args.train_split))
    train_df, val_df = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        safety_checker=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.to(dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    pipe.unet.to(dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    pipe.text_encoder.to(dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32)

    pipe = add_lora(pipe, rank=args.lora_rank, text_enc=args.train_text_encoder)

    def _print_lora_params(mdl, tag):
        total = 0
        trainable = 0
        any_lora = False
        for name, p in mdl.named_parameters():
            if "lora" in name.lower():
                any_lora = True
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
        return any_lora, trainable

    any_unet, _ = _print_lora_params(pipe.unet, "UNet")
    any_te, _ = (False, 0)
    if args.train_text_encoder and hasattr(pipe, "text_encoder"):
        any_te, _ = _print_lora_params(pipe.text_encoder, "TextEncoder")

    assert any_unet, "No UNet LoRA params found; check target_modules naming."
    if args.train_text_encoder:
        assert any_te, "No TextEncoder LoRA params found but --train_text_encoder set."

    # === Freeze base model ===
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # === Re-enable ONLY LoRA params ===

    for name, p in pipe.unet.named_parameters():
        if "lora" in name.lower():
            p.data = p.data.to(torch.float32)
            if p.grad is not None:
                p.grad = p.grad.to(torch.float32)
            p.requires_grad_(True)

    if args.train_text_encoder and hasattr(pipe, "text_encoder"):
        for name, p in pipe.text_encoder.named_parameters():
            if "lora" in name.lower():
                p.data = p.data.to(torch.float32)
                if p.grad is not None:
                    p.grad = p.grad.to(torch.float32)
                p.requires_grad_(True)
    if args.train_text_encoder and hasattr(pipe, "text_encoder"):
        for name, p in pipe.text_encoder.named_parameters():
            if "lora" in name.lower():
                p.requires_grad_(True)

    lora_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    if args.train_text_encoder and hasattr(pipe, "text_encoder"):
        lora_params += [p for p in pipe.text_encoder.parameters() if p.requires_grad]

    optim = torch.optim.AdamW(lora_params, lr=args.lr)

    train_set = EmoArtSet(train_df, tokenizer, size=args.resolution, center_crop=not args.no_center_crop, template=template, defaults=defaults)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.vae.to(device)
    pipe.unet.to(device)
    pipe.text_encoder.to(device)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))
    accum = args.grad_accum

    os.makedirs(args.out_dir, exist_ok=True)


    while global_step < args.max_steps:

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Training to {args.max_steps} steps")
        ):

            with torch.cuda.amp.autocast(enabled=(args.mixed_precision == "fp16")):

                pixel_values = batch["pixel_values"].to(device, dtype=pipe.unet.dtype, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)

                # --- TIMESTEPS ---
                bsz = pixel_values.shape[0]
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (bsz,), device=device
                ).long()

                # --- LATENT ENCODING ---
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                # --- FIX: SAMPLE NOISE IN LATENT SPACE ---
                noise = torch.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                enc_out = pipe.text_encoder(input_ids)
                enc = enc_out.last_hidden_state
                # --- UNET PREDICTION ---
                pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=enc,
                ).sample

                # --- TARGET ---
                if pipe.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif pipe.scheduler.config.prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(noisy_latents, noise, timesteps)
                else:
                    target = noise

                loss = (
                    torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="mean")
                    / accum
                )

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_every == 0:
                    print(f"step {global_step} | loss {loss.item() * accum:.4f}")

                # --- CHECKPOINT ---
                if args.ckpt_every and global_step % args.ckpt_every == 0:
                    ck = Path(args.out_dir) / f"checkpoint-{global_step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    pipe.save_lora_weights(str(ck))

                # --- CLEANUP ---
                if torch.cuda.is_available() and global_step % 50 == 0:
                    torch.cuda.empty_cache()

                if global_step >= args.max_steps:
                    break

    # --- FINAL SAVE ---
    unet_lora_layers = pipe.unet
    text_encoder_lora_layers = pipe.text_encoder if args.train_text_encoder and hasattr(pipe, "text_encoder") else None

    try:
        pipe.save_lora_weights(
            save_directory=str(Path(args.out_dir)),
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers
        )
        print(f"Saved LoRA weights successfully to {args.out_dir}")
    except Exception as e:
        print(f"Error saving LoRA weights: {e}")

# ------------- Argparse -------------
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    t.add_argument("--train_csv", type=str, default="train_emoart_paths.csv")
    t.add_argument("--resolution", type=int, default=384)
    t.add_argument("--batch_size", type=int, default=1)
    t.add_argument("--grad_accum", type=int, default=16)
    t.add_argument("--max_steps", type=int, default=300)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--warmup_steps", type=int, default=0)
    t.add_argument("--train_split", type=float, default=0.9)
    t.add_argument("--max_samples", type=int, default=None)
    t.add_argument("--mixed_precision", type=str, default="fp16")
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--lora_rank", type=int, default=16)
    t.add_argument("--train_text_encoder", action="store_true")
    t.add_argument("--no_center_crop", action="store_true")
    t.add_argument("--out_dir", type=str, default="outputs/emoart_lora")
    t.add_argument("--ckpt_every", type=int, default=1000)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--prompts", type=str, default="prompts.txt")
    t.add_argument("--num_workers", type=int, default=0)


    return p

if __name__ == "__main__":
    parser = build_parser()

    in_notebook = any(k in os.environ for k in ["JPY_PARENT_PID", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_GPU"])
    injected = os.environ.get("PIPELINE_ARGV", "")

    if injected:
        args = parser.parse_args(injected.split())
    elif in_notebook:
        args = parser.parse_args(["train"])
    else:
        args = parser.parse_args()

    train(args)

from diffusers import StableDiffusionPipeline
import torch
import os

# --- 1. Load base pipeline ---
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

# --- 2. Load Diffusers-compatible LoRA adapter ---
lora_dir = "outputs/emoart_lora"
pipe.load_lora_adapter(lora_dir)

# --- 3. Inference ---
prompt = "An expressive artwork, angry mood, negative valence, high arousal, deep reds and blacks, harsh stormy lighting, chaotic composition, in a painterly style, depicting a turbulent sea with crashing waves under a dark sky"
image = pipe(prompt, num_inference_steps=30, guidance_scale=8.0).images[0]
image.save("angry_storm.png")
