import torch
import os
from diffusers import FluxTransformer2DModel, FluxPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb


# MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_ID = "/tmp/cedric_sillaber_model_dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
OUTPUT_DIR = "flux_lora_output"


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
def main():

    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder_2 = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    tokenizer_2 = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FluxPipeline.from_pretrained(MODEL_ID, subfolder="scheduler", torch_dtype=torch.bfloat16).scheduler

    pipe = FluxPipeline(
        scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        vae=vae, transformer=transformer
    )
    pipe.enable_model_cpu_offload() #offloads model to CPU when not in use

    with torch.inference_mode():
        image = pipe(
            "A pic of a tyrolean pilot",
            num_inference_steps=50,
            guidance_scale=3.5,
            height=1024, width=1024
        ).images[0]
        image.save("test_before_train.png")

if __name__ == "__main__":
    main()