# Gen AI Project — Pic-AI-sso

This repository contains the full pipeline, data preparation scripts, and model utilities used for the **Gen AI Project** titled **Pic-AI-sso**, created by **Sarthak Kumar** and **Navansh Krishna Goswami**

## Project Overview
Pic-AI-sso is a generative AI project focused on **emotion‑conditioned image generation**. The system takes structured emotional metadata (emotion, valence, arousal, color scheme, lighting, composition, situation) and uses them to craft expressive prompts suitable for training text-to-image diffusion models.

## Repository Structure
```
├── pipeline.py     # Main project script
├── Images/               # Root directory containing dataset images
├── train_emoart_paths.csv# Auto-generated mapping of metadata to local image paths
└── README.md
```

## Features
- **Prompt Template Generator**: Converts emotion and scene metadata into a rich prompt suitable for generative models.
- **Image Directory Scanner**: Counts all images inside nested subdirectories in the `Images/` folder.
- **Dataset Loader (HuggingFace)**: Loads the `printblue/EmoArt-5k` dataset and maps each entry to its local image path.
- **CSV Builder**: Automatically constructs the `train_emoart_paths.csv` file used for training pipelines.
- **Training Data Preparation**: Ensures cleaned, normalized paths without RAM blowups.

## Requirements
Install all dependencies using:

```
pip install -r req.txt
```

## Usage
### 1. **Set Image Root**
Ensure that your dataset images exist at:
```
/kaggle/working/Images/
```
Modify `IMG_ROOT` in the script if required.

### 2. **Run the script**
```
python pipeline.py
```
This will:
- Count files
- Load EmoArt dataset
- Generate `train_emoart_paths.csv`
- Prepare prompts for training

## Dataset

**HuggingFace Dataset:** `printblue/EmoArt-5k`

Each row contains:
- `image_path`
- `emotion`
- `valence`
- `arousal`
- `color`
- `lighting`
- `composition`
- `situation`

## Prompt Template Format
Example template used:
```
an expressive artwork, {emotion} mood, {valence} valence, {arousal} arousal, {color} colors, {lighting} lighting, {composition}, in a painterly style, depicting {situation}
```
