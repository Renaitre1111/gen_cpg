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

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

config = {
    "dataset_name": "benjamin-paine/imagenet-1k-32x32",
    "clip_model_name": "openai/clip-vit-base-patch32",
    "output_dir": "diffusion_model/saved_models/",
    "image_size": 32,
    "batch_size": 1536,
    "num_epochs": 150,
    "learning_rate": 3e-4,
    "embedding_dim": 512,
    "mixed_precision": "bf16",
    "save_freq_epochs": 10,
    "num_workers": 8
}

def get_imagenet_label_names():
    builder = load_dataset_builder(config["dataset_name"])
    return builder.info.features["label"].names

def setup_dataset(label_names):
    dataset = load_dataset(config["dataset_name"], split="train", keep_in_memory=True)

    demo_split = dataset.train_test_split(
        train_size=0.1, 
        shuffle=True,
        stratify_by_column="label"
    )

    dataset = demo_split["train"]
    print(len(dataset))

    image_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def preprocess(examples):
        imgs = [image_transforms(img) for img in examples["image"]]
        labels = examples["label"]
        return {
            "pixel_values": torch.stack(imgs),
            "label": torch.tensor(labels, dtype=torch.long),
        }
    # dataset.set_transform(preprocess)
    processed_dataset = dataset.map(
        preprocess, 
        batched=True,
        remove_columns=["image"],
        num_proc=config["num_workers"],
    )

    processed_dataset = processed_dataset.with_format(
        type="torch",
        columns=["pixel_values", "label"]
    )
    print("accomplish processing")
    return processed_dataset

def get_label_embeddings(label_names, tokenizer, text_encoder, device):
    prompts = [f"a photo of a {name}" for name in label_names]
    text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
    return text_embeddings

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

    with torch.no_grad():
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids).last_hidden_state

    label_embeddings = get_label_embeddings(labels_names, tokenizer, text_encoder, accelerator.device)
    
    train_dataset = setup_dataset(labels_names)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"], persistent_workers=True, pin_memory=True)

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
    '''
    try:
        import xformers
        print("use Flash Attention")
        unet.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("Not found xformers")
    '''
    unet = torch.compile(unet, mode="reduce-overhead")
    # text_encoder = torch.compile(text_encoder, mode="max-autotune")

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

    label_embeddings = label_embeddings.to(
        accelerator.device,
        dtype=torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float16
    )

    text_encoder.to(accelerator.device)
    uncond_embeddings = uncond_embeddings.to(accelerator.device)

    cond_drop_prob = 0.1

    for epoch in range(config["num_epochs"]):
        unet.train()

        total_loss = 0.0
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}")

        for step, batch in enumerate(train_dataloader):
            images = batch["pixel_values"]
            label_ids = batch["label"].to(accelerator.device)

            with torch.no_grad():
                text_embeddings = label_embeddings[label_ids]

            bs = images.shape[0]
            mask = torch.rand(bs, device=accelerator.device) < cond_drop_prob
            mask_expand = mask.view(bs, 1, 1)

            final_embeddings = torch.where(
                mask_expand, 
                uncond_embeddings.expand_as(text_embeddings).to(text_embeddings.dtype), 
                text_embeddings
            )

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.autocast():
                noise_pred = unet(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=final_embeddings
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