import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig
from peft import PeftModel

import utils

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "cps/step_0700"
DEVICE = "cuda"

def generate_images(lora_path, quantization="4bit", save_dir=None, num_per_seed=None, seed=42, batch_number=1):
    assert quantization in ["4bit", "8bit"]
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    int8_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    quantization_config = nf4_config if quantization == "4bit" else int8_config
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )
    transformer = PeftModel.from_pretrained(transformer, lora_path)

    generator = torch.Generator(DEVICE).manual_seed(seed)
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    prompts = utils.get_prompts()
    if num_per_seed is None:
        for i,prompt in enumerate(prompts):
            images = pipe(
                prompt,
                num_inference_steps=38,
                guidance_scale=4.0,
                height=1024,
                width=1024,
                generator=generator,
                num_images_per_prompt=batch_number,
            ).images
            for k, image in enumerate(images):
                path = f"{save_dir}/output_{i}_{k}.png" if save_dir else f"output_{i}_{k}.png"
                image.save(path)
    else:
        for i,prompt in enumerate(prompts):
            for j in range(num_per_seed):
                images = pipe(
                    prompt,
                    num_inference_steps=40,
                    guidance_scale=4.0,
                    height=1024,
                    width=1024,
                    generator=generator,
                    num_images_per_prompt=batch_number
                ).images
                for k, image in enumerate(images):
                    path = f"{save_dir}/output_{i}_{j}_{k}.png" if save_dir else f"output_{i}_{j}_{k}.png"
                    image.save(path)
    del pipe, transformer


if __name__ == "__main__":

        save_dir = f"res/outputs/20251203/step_4000"
        os.makedirs(save_dir, exist_ok=True)

        # lora_path = f"cps/20253011_gpu11/step_0500"
        # lora_path = "cps/20253011_gpu11_too_high_lr/step_1500"
        lora_path = "20251201_flux_lora_output-1e-4/step_4000"
        print(f"{lora_path}")
        generate_images(
            lora_path=lora_path, save_dir=save_dir, num_per_seed=1, seed=123422, batch_number=4)


    # for i in range(15,17):
    #     save_dir = f"res/outputs/20253011_gpu11/step_{i}00"
    #     os.makedirs(save_dir, exist_ok=True)
    #     i_str = f"{i:02d}"
    #     lora_path = f"cps/20253011_gpu11/step_{i_str}00"
    #     print(f"{lora_path}")
    #     generate_images(lora_path=lora_path, save_dir=save_dir)

