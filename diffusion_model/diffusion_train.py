import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_dataset_builder
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator

config = {
    "dataset_name": "benjamin-paine/imagenet-1k-32x32",
    "clip_model_name": "openai/clip-vit-base-patch32",
    "output_dir": "diffusion_model/saved_models/",
    "image_size": 32,
    "batch_size": 2560,
    "num_epochs": 150,
    "learning_rate": 3e-4,
    "embedding_dim": 512,
    "mixed_precision": "bf16",
    "save_freq_epochs": 10,
    "num_workers": 32
}

def get_imagenet_label_names():
    builder = load_dataset_builder(config["dataset_name"])
    return builder.info.features["label"].names

def setup_dataset(label_names, tokenizer):
    dataset = load_dataset(config["dataset_name"], split="train")

    image_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def preprocess(examples):
        imgs = [image_transforms(img) for img in examples["image"]]
        label_ints = examples["label"]
        label_name = [label_names[int(i)] for i in label_ints]
        prompts = [f"a photo of a {name}" for name in label_name]

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": torch.stack(imgs),
            "input_ids": text_inputs.input_ids,
        }
    dataset.set_transform(preprocess)
    return dataset

def main():
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=1
    )

    tokenizer = CLIPTokenizer.from_pretrained(config["clip_model_name"])
    text_encoder = CLIPTextModel.from_pretrained(config["clip_model_name"])

    text_encoder.requires_grad_(False)
    text_encoder.eval()

    labels_names = get_imagenet_label_names()

    train_dataset = setup_dataset(labels_names, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    unet = UNet2DConditionModel(
        sample_size=config["image_size"],
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=config["embedding_dim"]
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * config["num_epochs"]),
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device)

    for epoch in range(config["num_epochs"]):
        unet.train()

        total_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}")

        for step, batch in enumerate(train_dataloader):
            images = batch["pixel_values"]
            text_input_ids = batch["input_ids"]

            with torch.no_grad():
                text_embeddings = text_encoder(text_input_ids).last_hidden_state

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=accelerator.device
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.autocast():
                noise_pred = unet(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        progress_bar.close()

        avg_loss = total_loss / len(train_dataloader)

        if accelerator.is_local_main_process:
            print(f"epoch {epoch + 1} loss: {avg_loss:.4f}")

        if accelerator.is_local_main_process:
            if (epoch + 1) % config["save_freq_epochs"] == 0 or (epoch + 1) == config["num_epochs"]:
                unwrapped_model = accelerator.unwrap_model(unet)
                epoch_output_dir = os.path.join(config["output_dir"], f"epoch_{epoch + 1}")
                os.makedirs(epoch_output_dir, exist_ok=True)
                unwrapped_model.save_pretrained(epoch_output_dir)
                noise_scheduler.save_config(epoch_output_dir)

if __name__ == "__main__":
    main()