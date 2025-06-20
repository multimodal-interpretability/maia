import torch
from typing import Optional
from PIL import Image
import gc
from transformers import T5EncoderModel
from diffusers import FluxPipeline, FluxTransformer2DModel

class FluxImageGenerator:
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        quant_id: str = "sayakpaul/flux.1-dev-nf4-pkg",
        text_encoder: Optional[T5EncoderModel] = None,
        device: str = "cuda:0"
    ):
        """Initialize FluxImageGenerator with model parameters."""
        self.model_id = model_id
        self.quant_id = quant_id
        self.device = device
        self.text_encoder = text_encoder or self._load_shared_text_encoder()
        self.pipeline = self._load_txt2img_pipeline()
        
    def _free_memory(self):
        """Free up CUDA memory and reset memory stats."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            
    def _load_shared_text_encoder(self) -> T5EncoderModel:
        """Load shared text encoder for both pipelines."""
        return T5EncoderModel.from_pretrained(
            self.quant_id,
            subfolder="text_encoder_2"
        ).to(self.device)
    
    def _load_txt2img_pipeline(self) -> FluxPipeline:
        """Load text-to-image pipeline."""
        # Load transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            self.quant_id,
            subfolder="transformer"
        ).to(self.device)
        
        # Initialize pipeline
        pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            text_encoder_2=self.text_encoder,
            transformer=transformer,
            torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload(device=self.device)
        pipeline.set_progress_bar_config(disable=True)
        
        return pipeline
    
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Generate image from text prompt."""
        # Free memory before generation
        # self._free_memory()
        
        # Encode prompt
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=256
            )
        
        # Set generator if seed provided
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        
        # Set default parameters
        params = {
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'output_type': 'pil',
            'height': height,
            'width': width,
            'generator': generator,
            **kwargs
        }
        
        # Generate image
        with torch.no_grad():
            images = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                **params
            ).images
        
        return images[0]