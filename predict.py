import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from cog import BasePredictor, Input, Path


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline....")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "dreambooth-output",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.safety_checker = None

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        num_outputs: int = Input(
            description="Number of images to output",
            choices=[1, 4],
            default=1,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[288, 640],
            default=640,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[288, 640],
            default=288,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt + " in the style of monkey island",
            num_images_per_prompt=num_outputs,
            width=width // 2,
            height=height // 2,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        images = [
            image.convert("RGB").resize([width, height], Image.Resampling.NEAREST)
            for image in output.images
        ]

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
