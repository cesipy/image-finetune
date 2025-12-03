import os

# not sure if they are even usefull, found some optims using those
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor-cache_cedric"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"

import torch

import utils

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig
from safetensors.torch import load_file


MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEVICE = "cuda"
LORA_PATH = "20251201_flux_lora_output-1e-4/step_3100"
SEED = 1219
SCALES_TO_TEST = [0.85]
BASE_OUTPUT_DIR = "res/outputs/20251202/step_2600_scale_test3/"
BATCH_SIZE = 6

def load_pipeline(quantization="4bit"):
    print("--- Loading Pipeline ---")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    int8_config = BitsAndBytesConfig(load_in_8bit=True)
    quantization_config = nf4_config if quantization == "4bit" else int8_config

    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()  # for batched VAE decode
    return pipe

def load_manual_lora(pipe, lora_path):
    """
    loading the lora manually in order to test the scale
    """
    if os.path.isdir(lora_path):
        weight_file = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(weight_file):
            weight_file = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    else:
        weight_file = lora_path

    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Could not find LoRA weights at {weight_file}")

    state_dict = load_file(weight_file)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model."):
            new_key = key.replace("base_model.model.", "transformer.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    pipe.load_lora_weights(new_state_dict, adapter_name="default")
    print("LoRA loaded.")

def run_inference_loop(pipe):
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.transformer.compile_repeated_blocks(fullgraph=True, mode="default")
    prompts = utils.get_prompts()

    for scale in SCALES_TO_TEST:
        save_dir = os.path.join(BASE_OUTPUT_DIR, f"scale_{scale}")
        os.makedirs(save_dir, exist_ok=True)

        generator = torch.Generator(DEVICE).manual_seed(SEED)
        for i, prompt in enumerate(prompts):

            images = pipe(

                prompt,
                num_inference_steps=30,
                guidance_scale=3.5,
                height=1024,
                width=1024,
                generator=generator,
                num_images_per_prompt=BATCH_SIZE,
                joint_attention_kwargs={"scale": scale},
            ).images

            for j, image in enumerate(images):
                image.save(os.path.join(save_dir, f"output_{i}_{j}.png"))

if __name__ == "__main__":
    pipeline = load_pipeline()
    load_manual_lora(pipeline, LORA_PATH)


    run_inference_loop(pipeline)