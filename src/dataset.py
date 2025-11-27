import os
from diffusers import FluxPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import torch; from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, captions_path: str):

        with open(captions_path, "r") as f:
            lines = f.readlines()

        t_data = [line.split("|;|") for line in lines]
        t_data = [(img_path.strip(), caption.strip()) for img_path, caption in t_data]
        self.data = t_data
        self.transforms = transforms.Compose( [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(self.data)

    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx):
        tup = self.data[idx]

        img = Image.open(tup[0]).convert("RGB")
        img = self.transforms(img)
        caption = tup[1]

        return {
            "image": img,
            "caption": caption
        }



MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "res/cache"
BATCH_SIZE = 1
DEVICE = "cuda"

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.bfloat16).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to(DEVICE)

    dataset = CustomDataset("res/images/captions.txt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


    for i, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE).to(torch.bfloat16)
        captions = batch["caption"] # List of strings

        with torch.no_grad():
            #vae for imgs
            latents = vae.encode(images).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            #clip
            text_inputs = tokenizer(
                captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
            )
            pooled_prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE)).pooler_output

            # t5
            text_inputs_2 = tokenizer_2(
                captions, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
            )
            prompt_embeds = text_encoder_2(text_inputs_2.input_ids.to(DEVICE))[0]

        save_path = os.path.join(CACHE_DIR, f"{i}.pt")
        torch.save({
            "latents": latents.cpu(),
            "prompt_embeds": prompt_embeds.cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.cpu()
        }, save_path)


if __name__ == "__main__":
    main()