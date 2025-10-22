import dataclasses
import gc
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from diffusers import (
    FluxKontextPipeline,
)
from diffusers.utils import load_image
from PIL import Image
from PIL.Image import Image as Img
from transformers import T5EncoderModel


@dataclass
class FluxKontextDevConfig:
    """Configuration for the FluxKontextDev pipeline."""

    transformer_ckpt: str = 'eramth/flux-kontext-4bit'
    pipe_id: str = 'black-forest-labs/FLUX.1-Kontext-dev'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.bfloat16
    max_sequence_length: int = 256
    text_encoder: T5EncoderModel | None = field(default=None, repr=False)


class FluxKontextDev:
    """
    FLUX.1-dev text-to-image pipeline with 4-bit components and proper batching
    - 4-bit (NF4) FLUX.Kontext transformer
    - 4-bit (NF4) T5-XXL encoder
    """

    def __init__(self, config: FluxKontextDevConfig | None = None, **kwargs) -> None:
        base_config = config or FluxKontextDevConfig()
        self.config = dataclasses.replace(base_config, **kwargs)
        self.pipe: FluxKontextPipeline
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Loads components, configures the pipeline, and applies optimizations."""
        # 1. Initialize Pipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            'eramth/flux-kontext-4bit', torch_dtype=torch.bfloat16
        )
        # 2. Apply Optimizations and Configuration Tweaks
        self.pipe.enable_model_cpu_offload(device=self.config.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.transformer.to(memory_format=torch.channels_last)
        self.pipe.vae.to(memory_format=torch.channels_last)
        self.pipe.vae.fuse_qkv_projections()

    @staticmethod
    def _free_memory() -> None:
        """Clears GPU cache and forces garbage collection."""
        gc.collect()
        torch.cuda.empty_cache()

    def _create_generator(self, seed: int = 42) -> torch.Generator | None:
        """Creates a torch.Generator object for reproducibility."""
        return torch.Generator(device=self.config.device).manual_seed(seed or 42)

    @staticmethod
    def _normalize_inputs(
        images: Img | str | Sequence[Img | str],
        prompts: str | Sequence[str],
    ) -> tuple[list[Img | str], list[str]]:
        """Ensures images and prompts are lists of equal length."""
        img_list = [images] if isinstance(images, (Img, str)) else list(images)
        prm_list = [prompts] if isinstance(prompts, str) else list(prompts)

        if len(prm_list) == 1 and len(img_list) > 1:
            prm_list *= len(img_list)  # Broadcast prompt

        if len(img_list) != len(prm_list):
            raise ValueError(
                f'Num. of images ({len(img_list)}) and prompts ({len(prm_list)}) must match.'
            )
        return img_list, prm_list

    @staticmethod
    def _prepare_image(image: str | Img) -> Img:
        """Loads an image from a path or ensures it's in RGB format."""
        if isinstance(image, str):
            return load_image(image).convert('RGB')
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        raise TypeError(f'Image must be a path or PIL.Image, got {type(image)}')

    @torch.no_grad()
    def generate(
        self,
        images: Img | str | Sequence[Img | str],
        instructions: str | Sequence[str],
        guidance_scale: float = 2.5,
        num_inference_steps: int = 15,
        seed: int | None = None,
        batch_size: int = 4,
        **kwargs: object,
    ) -> list[Img]:
        """
        Image-to-Image generation with proper batching.

        Args:
            images: A single image or a list of images (PIL Image or path string).
            prompts: A single prompt (str) or a list of prompts.
            guidance_scale: The scale for classifier-free guidance.
            num_inference_steps: The number of denoising steps.
            seed: An optional seed for reproducible generation.
            batch_size: The number of images to process in a single batch.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            A list of generated PIL Images.
        """
        # 1. Normalize and validate inputs
        images, instructions = self._normalize_inputs(images, instructions)

        # 2. Prepare for generation
        self._free_memory()
        results: list[Img] = []
        generator = self._create_generator(seed)

        # 3. Process in batches
        for i in range(0, len(images), batch_size):
            # Prepare a slice of the inputs for the current batch
            image_batch_input = images[i : i + batch_size]
            instruction_batch = instructions[i : i + batch_size]

            # Pre-process images to PIL format
            image_batch_pil = [self._prepare_image(img) for img in image_batch_input]

            # 2) Pre-encode prompts like you do for regular Flux
            instruction_embeds, pooled_instruction_embeds, _ = self.pipe.encode_prompt(
                prompt=instruction_batch,
                prompt_2=None,
                max_sequence_length=self.config.max_sequence_length,
            )

            # 3) Run the pipeline. If multiple images per call don’t work on your build,
            batch = self.pipe(
                image=image_batch_pil,  # try list batching first
                prompt_embeds=instruction_embeds,
                pooled_prompt_embeds=pooled_instruction_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type='pil',
                **kwargs,
            ).images

            results.extend(batch)

        del instruction_embeds, pooled_instruction_embeds
        self._free_memory()
        return results

    def __call__(
        self, instructions: str | Sequence[str], images: Sequence[Img], **kwargs
    ) -> list[Img]:
        return self.generate(instructions=instructions, images=images, **kwargs)
