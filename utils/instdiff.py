from __future__ import annotations

import os
import math
import random
import sys
from argparse import ArgumentParser
from typing import Tuple, Dict, Optional, List
from pathlib import Path

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast


# Get absolute paths
current_dir = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd()
instruct_dir = current_dir / "InstructDiffusion"
stable_diff_dir = instruct_dir / "stable_diffusion"

# Add all necessary paths
sys.path.extend([
    str(current_dir),                # For local imports
    str(stable_diff_dir),           # For ldm module
    str(instruct_dir),              # For InstructDiffusion modules
    str(stable_diff_dir / "src"),   # For additional dependencies
    str(stable_diff_dir / "taming"),  # For taming module
])

from ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    """
    Classifier-Free Guidance Denoiser module that combines conditional and unconditional denoising.
    
    This module applies both text and image conditioning during the denoising process.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the CFG Denoiser.

        Args:
            model: The base denoising model to wrap
        """
        super().__init__()
        self.inner_model = model

    def forward(self, z: torch.Tensor, sigma: torch.Tensor, 
                cond: Dict[str, list], uncond: Dict[str, list], 
                text_cfg_scale: float, image_cfg_scale: float) -> torch.Tensor:
        """
        Forward pass of the CFG Denoiser.

        Args:
            z: Input noise tensor
            sigma: Noise level tensor
            cond: Conditional inputs (text and image)
            uncond: Unconditional inputs
            text_cfg_scale: Text conditioning scale factor
            image_cfg_scale: Image conditioning scale factor

        Returns:
            Denoised output tensor combining conditional and unconditional predictions
        """
        # Repeat input for conditional, unconditional, and image branches
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        
        # Combine conditional inputs
        cfg_cond = {
            "c_crossattn": [torch.cat([
                cond["c_crossattn"][0], 
                uncond["c_crossattn"][0], 
                cond["c_crossattn"][0]
            ])],
            "c_concat": [torch.cat([
                cond["c_concat"][0], 
                cond["c_concat"][0], 
                uncond["c_concat"][0]
            ])],
        }
        
        # Get predictions for all branches
        out_cond, out_img_cond, out_txt_cond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond
        ).chunk(3)
        
        # Combine predictions with scaling factors
        return (0.5 * (out_img_cond + out_txt_cond) + 
                text_cfg_scale * (out_cond - out_img_cond) + 
                image_cfg_scale * (out_cond - out_txt_cond))


class InstructDiffusion:
    """
    Main class for running instruction-based image editing using diffusion models.
    
    This class handles model loading, image preprocessing, and generation of edited images
    based on text instructions. Supports batch processing of multiple images.
    """
    def __init__(
        self,
        config_path: str = "InstructDiffusion/configs/instruct_diffusion.yaml",
        model_path: str = "InstructDiffusion/checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt",
        device: str = "cuda",
        num_steps: int = 100,
        text_cfg_scale: float = 5.0,
        image_cfg_scale: float = 1.25,
        batch_size: int = 4  # Added batch size parameter
    ):
        """
        Initialize the InstructDiffusion model.

        Args:
            config_path: Path to the model configuration file
            model_path: Path to the model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            num_steps: Number of denoising steps
            text_cfg_scale: Text conditioning scale factor
            image_cfg_scale: Image conditioning scale factor
            batch_size: Maximum number of images to process simultaneously
        """
        self.device = device
        self.num_steps = num_steps
        self.text_cfg_scale = text_cfg_scale
        self.image_cfg_scale = image_cfg_scale
        self.batch_size = batch_size
        
        # Load and initialize model
        self.model = self._load_model(config_path, model_path)
        self.model_wrap = K.external.CompVisDenoiser(self.model, device=self.device)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.model.cond_stage_model.to(self.device)
        self.model.cond_stage_model.transformer.to(self.device)
        self.model.cond_stage_model.device = self.device
        self.null_token = self.model.get_learned_conditioning([""])

    def _load_model(self, config_path: str, model_path: str) -> nn.Module:
        """
        Load the model from configuration and checkpoint.

        Args:
            config_path: Path to model configuration
            model_path: Path to model checkpoint

        Returns:
            Loaded and initialized model
        """
        config = OmegaConf.load(config_path)
        model = instantiate_from_config(config.model)
        
        print(f"Loading model from {model_path}")
        pl_sd = torch.load(model_path, map_location="cpu")
        if 'state_dict' in pl_sd:
            pl_sd = pl_sd['state_dict']
        
        missing, unexpected = model.load_state_dict(pl_sd, strict=False)
        print(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
        
        return model.eval().to(self.device)

    def _preprocess_images(self, images: List[Image.Image]) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
        """
        Preprocess a batch of input images to correct size and format.

        Args:
            images: List of input PIL Images

        Returns:
            Tuple of (preprocessed images, list of original dimensions)
        """
        processed_images = []
        original_sizes = []
        
        for image in images:
            width, height = image.size
            
            # Calculate resize factor to fit within 512 pixels while maintaining aspect ratio
            factor = 512 / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            
            # Calculate new dimensions divisible by 64
            width_resize = int((width * factor) // 64) * 64
            height_resize = int((height * factor) // 64) * 64
            
            # Resize image
            resized_img = ImageOps.fit(
                image, (width_resize, height_resize), 
                method=Image.Resampling.LANCZOS
            )
            
            processed_images.append(resized_img)
            original_sizes.append((width, height))
        
        return processed_images, original_sizes

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode multiple images to latent space.

        Args:
            images: List of PIL Images to encode

        Returns:
            Batch of encoded image tensors
        """
        # Convert PIL images to tensor
        img_tensors = []
        for img in images:
            img_tensor = 2 * torch.tensor(np.array(img)).float() / 255 - 1
            img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")
            img_tensors.append(img_tensor)
        
        # Stack tensors into batch
        batch_tensor = torch.cat(img_tensors, dim=0).to(self.device)
        
        # Encode to latent space
        return self.model.encode_first_stage(batch_tensor).mode()

    def _decode_images(self, z: torch.Tensor) -> List[Image.Image]:
        """
        Decode batch of latent tensors to PIL Images.

        Args:
            z: Batch of latent tensors to decode

        Returns:
            List of decoded PIL Images
        """
        x = self.model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "b c h w -> b h w c")
        
        return [Image.fromarray(img.type(torch.uint8).cpu().numpy())
                for img in x]

    def _process_batch(
        self, 
        images: List[Image.Image], 
        instructions: List[str],
        original_sizes: List[Tuple[int, int]]
    ) -> List[Image.Image]:
        """
        Process a single batch of images. Handles padding if batch is not full.

        Args:
            images: List of preprocessed images
            instructions: List of editing instructions
            original_sizes: List of original image dimensions

        Returns:
            List of edited images
        """
        # Handle batch padding if needed
        num_images = len(images)
        if num_images < self.batch_size:
            # Pad with copies of the last image and instruction
            pad_images = images + [images[-1]] * (self.batch_size - num_images)
            pad_instructions = instructions + [instructions[-1]] * (self.batch_size - num_images)
        else:
            pad_images = images
            pad_instructions = instructions
        with torch.no_grad(), autocast(device_type="cuda"):
            # Prepare conditional inputs
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning(pad_instructions)],
                "c_concat": [self._encode_images(pad_images)]
            }
            
            # Prepare unconditional inputs
            uncond = {
                "c_crossattn": [self.model.get_learned_conditioning([""] * self.batch_size)],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])]
            }

            # Cast everything to float32
            cond_latents = cond["c_concat"][0].float()
            cond["c_concat"][0] = cond_latents
            uncond["c_concat"][0] = uncond["c_concat"][0].float()
            
            # Setup sampling parameters
            sigmas = self.model_wrap.get_sigmas(self.num_steps).float()
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": self.text_cfg_scale,
                "image_cfg_scale": self.image_cfg_scale,
            }
            
            # Generate edited images
            torch.manual_seed(random.randint(0, 2**32 - 1))
            z = torch.randn(cond_latents.shape, device=cond_latents.device, dtype=torch.float32) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(
                self.model_wrap_cfg, z, sigmas, extra_args=extra_args
            )
            
            # Decode and post-process
            edited_images = self._decode_images(z)
            
            # Remove padding and resize back to original dimensions
            edited_images = edited_images[:num_images]  # Remove padding
            return [
                ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
                for img, size in zip(edited_images, original_sizes)
            ]

    def __call__(self, instructions: List[str], images: List[Image.Image]) -> List[Image.Image]:
        """
        Generate edited images based on instructions.

        Args:
            instructions: List of text instructions describing the desired edits
            images: List of input PIL Images to edit

        Returns:
            List of edited PIL Images

        Raises:
            ValueError: If number of images and instructions don't match or if images list is empty
        """
        if not images:
            raise ValueError("Images list cannot be empty")
        if len(images) != len(instructions):
            raise ValueError("Number of images must match number of instructions")
        
        if not images:
            return []
        
        # Preprocess all images
        processed_images, original_sizes = self._preprocess_images(images)
        
        # Process in batches
        edited_images = []
        for i in range(0, len(images), self.batch_size):
            batch_images = processed_images[i:i + self.batch_size]
            batch_instructions = instructions[i:i + self.batch_size]
            batch_sizes = original_sizes[i:i + self.batch_size]
            
            edited_batch = self._process_batch(
                batch_images, batch_instructions, batch_sizes
            )
            edited_images.extend(edited_batch)
        
        return edited_images