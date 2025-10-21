# utils/flux_dev.py
import dataclasses
import gc
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from PIL.Image import Image as Img
from transformers import T5EncoderModel


@dataclass
class FluxDevConfig:
    """Configuration for the FLUX.1-dev text-to-image pipeline."""

    transformer_ckpt: str = 'hf-internal-testing/flux.1-dev-nf4-pkg'
    text_encoder_ckpt: str = 'WaveCut/google-flan-t5-xxl-encoder_bnb-nf4'
    pipe_id: str = 'black-forest-labs/FLUX.1-dev'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float16
    max_sequence_length: int = 256
    text_encoder: T5EncoderModel | None = field(default=None, repr=False)


class FluxDev:
    """
    FLUX.1-dev text-to-image pipeline with 4-bit components and proper batching.
    - 4-bit (NF4) FLUX.1-dev transformer
    - 4-bit (NF4) T5-XXL encoder
    """

    def __init__(self, config: FluxDevConfig | None = None, **kwargs) -> None:
        base_config = config or FluxDevConfig()
        self.config = dataclasses.replace(base_config, **kwargs)
        self.pipe: FluxPipeline
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Loads components, configures the pipeline, and applies optimizations."""
        transformer = FluxTransformer2DModel.from_pretrained(
            self.config.transformer_ckpt, subfolder='transformer'
        )

        text_encoder = self.config.text_encoder or T5EncoderModel.from_pretrained(
            self.config.text_encoder_ckpt,
            torch_dtype=self.config.dtype,
        )

        # 2. Initialize Pipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.config.pipe_id,
            transformer=transformer,
            text_encoder_2=text_encoder,
            torch_dtype=self.config.dtype,
        )

        # 3. Apply Optimizations and Configuration Tweaks
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

    @torch.no_grad()
    def generate(
        self,
        prompts: str | Sequence[str],
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 5.5,
        num_inference_steps: int = 25,
        seed: int | None = None,
        batch_size: int = 4,
        **kwargs: object,
    ) -> list[Img]:
        """
        Text-to-Image generation with proper batching.

        Args:
            prompts: A single prompt (str) or a list of prompts.
            height/width: Output image size.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of denoising steps.
            seed: Optional seed for reproducibility.
            batch_size: Number of prompts to process per batch.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            A list of generated PIL Images.
        """
        self._free_memory()
        results: list[Img] = []
        generator = self._create_generator(seed)

        if isinstance(prompts, str):
            prompts = [prompts]

        for i in range(0, len(prompts), batch_size):
            prompt_batch = prompts[i : i + batch_size]

            prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt_batch,
                prompt_2=None,
                max_sequence_length=self.config.max_sequence_length,
            )

            batch_results = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type='pil',
                **kwargs,
            ).images
            results.extend(batch_results)

        del prompt_embeds, pooled_prompt_embeds
        self._free_memory()
        return results

    def __call__(self, prompts: str | Sequence[str], **kwargs) -> list[Img]:
        return self.generate(prompts, **kwargs)
