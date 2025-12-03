import os
import torch
from torch.utils.data import DataLoader, Dataset; from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
import tqdm

MODEL_ID = "black-forest-labs/FLUX.1-dev"
CACHE_DIR = "res/cache"
OUTPUT_DIR = "20251201_flux_lora_output-1e-4"
STEPS = 4000
LEARNING_RATE = 1e-4
BATCH_SIZE = 3
DEVICE = "cuda"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def sample(transformer, step, output_dir, prompt="a photo of gkqz man"):
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            height=512,
            width=512,
        ).images[0]

    image.save(os.path.join(output_dir, f"sample_{step:04d}.png"))
    print(f"Saved sample at step {step}")

    del pipe
    torch.cuda.empty_cache()



def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents

def _prepare_ids(batch_size, height, width, text_len, device):
    img_ids = torch.zeros(height // 2, width // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_ids = img_ids.reshape(-1, 3).to(device)
    txt_ids = torch.zeros(text_len, 3).to(device)
    return img_ids, txt_ids


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu")

def collate_fn(batch):
    latents = torch.stack([item["latents"] for item in batch]).squeeze(1)
    prompt_embeds = torch.stack([item["prompt_embeds"] for item in batch]).squeeze(1)
    pooled_prompt_embeds = torch.stack([item["pooled_prompt_embeds"] for item in batch]).squeeze(1)
    return latents, prompt_embeds, pooled_prompt_embeds

def train(resume_path=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # to get a quantized model
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    int8_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )


    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=nf4_config,
        # quantization_config=int8_config,
        torch_dtype=torch.bfloat16
    )

    transformer.enable_gradient_checkpointing()

    # force gradients on the first layer (x_embedder)
    transformer.x_embedder.register_forward_hook(make_inputs_require_grad)

    if resume_path:
        transformer = PeftModel.from_pretrained(
            transformer,
            resume_path,
            is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0",
                "add_k_proj", "add_q_proj", "add_v_proj",
            ],
            bias="none"
        )
        transformer = get_peft_model(transformer, lora_config)
    transformer = torch.compile(transformer)
    transformer.train()

    optimizer = bnb.optim.AdamW8bit(
        transformer.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    dataset = CachedDataset(CACHE_DIR)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("Starting training loop...")
    global_step = 0

    total_loss = 0
    counter = 0
    while global_step < STEPS:
        for i, batch in tqdm.tqdm(enumerate(dl), total=len(dl), leave=False):
            if global_step >= STEPS: break

            latents, prompt_embeds, pooled_prompt_embeds = [x.to(DEVICE, dtype=torch.bfloat16) for x in batch]

            noise = torch.randn_like(latents).to(DEVICE)
            bsz = latents.shape[0]
            # t = torch.sigmoid(torch.randn((bsz,), device=DEVICE))
            t = torch.rand((bsz,), device=DEVICE)

            t_expanded = t.view(bsz, 1, 1, 1)
            noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
            target = noise - latents

            # Packing & IDs
            h, w = latents.shape[2], latents.shape[3]
            packed_noisy_latents = _pack_latents(noisy_latents, bsz, 16, h, w)
            packed_target = _pack_latents(target, bsz, 16, h, w)
            img_ids, txt_ids = _prepare_ids(bsz, h, w, prompt_embeds.shape[1], DEVICE)

            model_pred = transformer(
                hidden_states=packed_noisy_latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=torch.tensor([1.0], device=DEVICE, dtype=torch.bfloat16).expand(bsz),
                return_dict=False
            )[0]

            loss = torch.nn.functional.mse_loss(model_pred.float(), packed_target.float(), reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step();#scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            counter += 1
            global_step += 1


            if (global_step ) % 100 == 0:
                transformer.save_pretrained(f"{OUTPUT_DIR}/step_{global_step:04d}")

        avg_loss = total_loss / counter
        print(f"Step {global_step}/{STEPS}, Loss: {avg_loss:.4f}")
        total_loss = 0
        counter = 0


    transformer.save_pretrained(OUTPUT_DIR)



if __name__ == "__main__":
    # train()
    path = "cps/20250112_gpu10-1e-4/step_2000"
    train(resume_path=path)