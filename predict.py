# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import torch
from PIL import Image
from typing import List
from torchvision import transforms
from diffusers import DiffusionPipeline
from weights_downloader import WeightsDownloader
from clip_interrogator import Config, Interrogator


MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE="model-cache"
MODELS = [
    {
      'name': 'diffusion-pipeline',
      'dest': 'model-cache/diffusion-pipeline',
      'src': "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar",
    },
    {
      'name': 'blip-large',
      'dest': 'model-cache/models--Salesforce--blip-image-captioning-large',
      'src': 'https://weights.replicate.delivery/default/clip-vit-bigg-14-laion2b-39b-b160k/salesforce-blip-image-captioning-large.tar'
    },
    {
      'name': 'ViT-bigG-14/laion2b_s39b_b160k',
      'dest': 'model-cache/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k',
      'src': "https://weights.replicate.delivery/default/clip-vit-bigg-14-laion2b-39b-b160k/laion-clip-vit-bigg-14-laion2b-39b-b160k.tar"
    }
]


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image

def load_and_process_image(pil_image):
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(pil_image)
    image = image.unsqueeze(0).half()
    return image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        for model in MODELS:
            WeightsDownloader.download_if_not_exists(model.get('src'), model.get('dest'))
        pipe = DiffusionPipeline.from_pretrained(
            'model-cache/diffusion-pipeline',
            custom_pipeline="pipeline_demofusion_sdxl.py",
            custom_revision="main",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe = pipe.to("cuda")
        self.ci = Interrogator(Config(clip_model_name="ViT-bigG-14/laion2b_s39b_b160k", clip_model_path=MODEL_CACHE, download_cache=False))

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image",
        ),
        scale: int = Input(
            description="Scale factor for input image",
            default=2,
            choices=[1, 2, 4]
        ),
        prompt: str = Input(
            description="Input prompt",
            default="A high resolution photo",
        ),
        auto_prompt: bool = Input(
            description="Select to use auto-generated CLIP prompt instead of using the above custom prompt",
            default=False,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=8.5
        ),
        view_batch_size: int = Input(
            description="The batch size for multiple denoising paths",
            default=16,
        ),
        stride: int = Input(
            description="The stride of moving local patches",
            default=64,
        ),
        cosine_scale_1: float = Input(
            description="Control the strength of skip-residual",
            default=3.0,
        ),
        cosine_scale_2: float = Input(
            description="Control the strength of dilated sampling",
            default=1.0,
        ),
        cosine_scale_3: float = Input(
            description="Control the strength of the Gaussian filter",
            default=1.0,
        ),
        sigma: float = Input(
            description="The standard value of the Gaussian filter",
            default=0.8,
        ),
        multi_decoder: bool = Input(
            description="Use multiple decoders",
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)
        pil_image  = Image.open(image)
        if auto_prompt:
            prompt = self.ci.interrogate(pil_image)
        print("Prompt: ", prompt)

        padded_image = pad_image(pil_image).resize((1024, 1024)).convert("RGB")
        image_lr = load_and_process_image(padded_image).to('cuda')

        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image_lr=image_lr,
            width=1024 * scale,
            height=1024 * scale,
            view_batch_size=view_batch_size,
            stride=stride,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cosine_scale_1=cosine_scale_1,
            cosine_scale_2=cosine_scale_2,
            cosine_scale_3=cosine_scale_3,
            sigma=sigma,
            multi_decoder=multi_decoder,
            show_image=False,
        )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths[-1]