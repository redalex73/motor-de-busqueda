import ray
import torch
import open_clip
from PIL import Image
from torchvision import transforms

@ray.remote
def compute_image_embedding(image_path):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return image_path, embedding.cpu().numpy()

@ray.remote
def compute_text_embedding(text_path):
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    with open(text_path, 'r') as f:
        text = f.read()
    tokenized = tokenizer([text])
    with torch.no_grad():
        embedding = model.encode_text(tokenized)
    return text_path, embedding.cpu().numpy()
