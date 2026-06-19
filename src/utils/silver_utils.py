
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()


prompts = [
    "a handcart in a warehouse",
    "a person pushing a handcart",
    "a luggage trolley cart",
    "a shopping cart in a supermarket"
]

text = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


def clip_score(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)
        score = similarity.max().item()

    return score
