import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig
from peft import PeftModel

# MODEL_ID = "/tmp/cedric_sillaber_model_dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
# LORA_PATH = "/tmp/cedric_sillaber/lora_output/step_1000"  # or step_0500, etc.
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "cps/step_0700"
DEVICE = "cuda"

def generate_images(lora_path, save_dir=None):
    # nf4_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True
    # )
    int8_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        # quantization_config=nf4_config,
        quantization_config=int8_config,
        torch_dtype=torch.bfloat16
    )
    transformer = PeftModel.from_pretrained(transformer, lora_path)

    print("Loading pipeline...")
    generator = torch.Generator(DEVICE).manual_seed(42)
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    prompts = [
        "a professional photo of gkqz man in a navy suit, studio lighting, sharp focus",
        "a photo of gkqz man on a mountain, golden hour, ",
        "a photo of gkqz man taking a selfie in a mirror in a barber shop",
        "a photo of gkqz man with a woman holding up glasses of beer",
        "a photo of gkqz man as an astronaut on the moon",
        "a photo of gkqz man wearing a chef hat in a kitchen",
        "a photo of gkqz man at the beach with sunglasses",
        "a photo of gkqz man in a leather jacket on a motorcycle",
        "a portrait of gkqz man in dramatic lighting",
        "a photo of gkqz man reading a book in a library",
        "a photo of gkqz man in a pink hoodie sitting in a car.",
        "a photo of gkqz man sitting in the cockpit of a small plane wearing headphones."
    ]

    for i,prompt in enumerate(prompts):
        image = pipe(
            prompt,
            num_inference_steps=38,
            guidance_scale=4.0,
            height=1024,
            width=1024,
            generator=generator
        ).images[0]
        path = f"{save_dir}/output_{i}.png" if save_dir else f"output_{i}.png"
        image.save(path)


if __name__ == "__main__":
    for i in range(1, 8):
        save_dir = f"outputs/step_{i}00"
        os.makedirs(save_dir, exist_ok=True)
        lora_path = f"cps/step_0{i}00"
        generate_images(lora_path=lora_path, save_dir=save_dir)