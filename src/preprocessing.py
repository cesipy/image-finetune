import torch
from PIL import Image
from torchvision import transforms

from transformers import AutoProcessor, AutoModelForCausalLM


import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


input_path = "res/images_lora"
output_path = "res/images"

def resize_crop_images(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)
    for i, filename in enumerate(os.listdir(input_path)):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            c_path = os.path.join(input_path, filename)
            img = Image.open(c_path).convert("RGB")
            crop = transforms.Compose([
                transforms.CenterCrop(min(img.height, img.width)),
                transforms.Resize((1024, 1024))
            ])
            img = crop(img)
            print(f"verifying shape: {img.size}")
            img.save(os.path.join(output_path, f"img_{i}.jpg"))


def create_caption(img_path, model, processor):
    image = Image.open(img_path).convert("RGB")
    # "<MORE_DETAILED_CAPTION>" is a specific task token for Florence-2
    prompt = "<CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=1,
        do_sample=False,
        use_cache=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Florence outputs usually look like: "<s><MORE_DETAILED_CAPTION>The image shows..."
    parsed_caption = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )[prompt]
    return parsed_caption


def create_captions(input_dir: str, trigger_word="gkqz"):
    img_cap_pairs = []
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(input_dir, file)
            caption = create_caption(img_path, model=model, processor=processor)
            caption = f"{trigger_word} {caption}"
            print(f"Caption for {file}: {caption}")
            img_cap_pairs.append((img_path, caption))
    print(f"extracted {len(img_cap_pairs)} captions")

    with open(os.path.join(input_dir, "captions.txt"), "w") as f:
        for img_path, caption in img_cap_pairs:
            f.write(f"{img_path}|;|{caption}\n")


def main():
    resize_crop_images(input_path, output_path)
    create_captions(output_path)

if __name__ == "__main__":
    main()

