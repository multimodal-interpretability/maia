# Standard library imports
import base64
import math
import os
import sys
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import baukit
import clip
import numpy as np
import openai
import requests
import timm
import torch
import torch.nn.functional as F
from baukit import Trace
from diffusers import (
    AutoPipelineForText2Image,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import models, transforms

# Local imports
from call_agent import ask_agent
from netdissect.imgviz import ImageVisualizer

# TODO: Remove these sys.path.append calls and properly structure the project
#sys.path.append('./synthetic-neurons-dataset/')
#sys.path.append('./synthetic-neurons-dataset/Grounded-Segment-Anything/')
#sys.path.append('./netdissect/')


class System:
    def __init__(self, unit_dict: Dict[str, Dict[str, List[int]]], thresholds: Dict[str, Dict[str, Dict[int, float]]], device: str):
        self.unit_dict = unit_dict
        self.thresholds = thresholds
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.current_model: Optional[str] = None
        self.current_layer: Optional[str] = None
        self.current_neuron: Optional[int] = None
        self.model_dict: Dict[str, 'ModelInfoWrapper'] = {}
        self.threshold: float = 0
        self.model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[callable] = None

        self._initialize_models()
        self._select_initial_neuron()

    def _initialize_models(self):
        model_names = self.unit_dict.keys()
        self.model_dict = {model_name: self.ModelInfoWrapper(model_name, self.device) for model_name in model_names}

    def _select_initial_neuron(self):
        if self.unit_dict:
            model_name = next(iter(self.unit_dict))
            layer = next(iter(self.unit_dict[model_name]))
            neuron_num = self.unit_dict[model_name][layer][0]
            self.select_neuron(model_name, layer, neuron_num)
        else:
            raise ValueError("unit_dict is empty")

    def select_neuron(self, model_name: str, layer: str, neuron_num: int):
        model_wrapper = self.model_dict[model_name]
        self.current_model = model_name
        self.model = model_wrapper.model
        self.preprocess = model_wrapper.preprocess
        self.current_layer = layer
        self.current_neuron = neuron_num
        if self.thresholds:
            self.threshold = self.thresholds[self.current_model][self.current_layer][self.current_neuron]
        else:
            self.threshold = 0

    @staticmethod
    def spatialize_vit_mlp(hiddens: torch.Tensor) -> torch.Tensor:
        batch_size, n_patches, n_units = hiddens.shape
        hiddens = hiddens[:, 1:]
        n_patches -= 1
        size = math.isqrt(n_patches)
        assert size**2 == n_patches
        return hiddens.permute(0, 2, 1).reshape(batch_size, n_units, size, size)

    def calc_activations(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with Trace(self.model, self.current_layer) as ret:
            _ = self.model(image)
            hiddens = ret.output

        if "dino" in self.current_model:
            hiddens = self.spatialize_vit_mlp(hiddens)

        batch_size, channels, *_ = hiddens.shape
        activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
        neuron_activation_map = hiddens[:, self.current_neuron, :, :]
        return pooled[:, self.current_neuron], neuron_activation_map

    def calc_class(self, image: torch.Tensor) -> Tuple[float, torch.Tensor]:
        logits = self.model(image)
        prob = F.softmax(logits, dim=1)
        image_class = torch.argmax(logits[0])
        activation = prob[0][image_class]
        return activation.item(), image

    def call_neuron(self, image_list: List[Image.Image]) -> Tuple[List[float], List[str]]:
        activation_list = []
        masked_images_list = []
        for image in image_list:
            if image is None:
                activation_list.append(None)
                masked_images_list.append(None)
            else:
                preprocessed_image = self.preprocess_images(image)
                if self.current_layer == 'last':
                    acts, _ = self.calc_class(preprocessed_image)
                    activation_list.append(acts)
                    masked_images_list.append(None)
                else:
                    acts, masks = self.calc_activations(preprocessed_image)
                    ind = torch.argmax(acts).item()
                    masked_image = generate_masked_image(preprocessed_image[ind], masks[ind], "./temp.png", self.threshold)
                    activation_list.append(acts[ind].item())
                    masked_images_list.append(masked_image)
        return activation_list, masked_images_list

    def preprocess_images(self, images):
        if isinstance(images, list):
            return torch.stack([self.preprocess(img).to(self.device) for img in images])
        else:
            return self.preprocess(images).unsqueeze(0).to(self.device)

    class ModelInfoWrapper:
        def __init__(self, model_name: str, device: torch.device):
            self.model_name = model_name
            self.device = device
            if 'dino' in model_name or 'resnet' in model_name:
                self.preprocess = self._preprocess_imagenet
            self.model = self._load_model(model_name)

        def _load_model(self, model_name: str) -> torch.nn.Module:
            if model_name == 'resnet152':
                model = models.resnet152(weights='IMAGENET1K_V1').to(self.device).eval()
            elif model_name == 'dino_vits8':
                model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(self.device).eval()
            elif model_name == "clip-RN50":
                full_model, preprocess = clip.load('RN50')
                model = full_model.visual.to(self.device).eval()
                self.preprocess = preprocess
            elif model_name == "clip-ViT-B32":
                full_model, preprocess = clip.load('ViT-B/32')
                model = full_model.visual.to(self.device).eval()
                self.preprocess = preprocess
            elif "gelu" in model_name:
                check_path = {
                    "finetune_resnet_gelu": "/data/vision/torralba/scratch/adrianr/input_norm/eccv_outputs/finetune_resnet_gelu/2024-02-01_16-32-09/checkpoint-3.pth.tar",
                    "advtrain_resnet_gelu": "/data/vision/torralba/scratch/adrianr/input_norm/eccv_outputs/advtrain_resnet_gelu/2024-02-01_16-32-09/checkpoint-3.pth.tar",
                    "gradnorm_resnet_gelu": "/data/vision/torralba/scratch/adrianr/input_norm/eccv_outputs/gradnorm_resnet_gelu/2024-02-03_22-07-28/snapshots/snapshot-49-5003.pth.tar"
                }[model_name]
                model = timm.models.create_model("resnet50", checkpoint_path=check_path, pretrained=True).to(self.device).eval()
                self._replace_layers(model, torch.nn.ReLU, torch.nn.GELU)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            return model

        def _preprocess_imagenet(self, image, normalize=True, im_size=224):
            if normalize:
                preprocess = transforms.Compose([
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                preprocess = transforms.Compose([
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                ])
            return preprocess(image)

        def _replace_layers(self, model, old, new):
            for n, module in model.named_children():
                if len(list(module.children())) > 0:
                    self._replace_layers(module, old, new)
                if isinstance(module, old):
                    setattr(model, n, new())

class Tools(BaseModel):
    path2save: str
    device: str
    dataset_exemplars: Optional['DatasetExemplars'] = None
    images_per_prompt: int = 10
    text2image_model_name: str = 'sd'
    
    text2image_model: Any
    p2p_model: Any
    experiment_log: List[Dict] = []
    im_size: int = 224
    activation_threshold: float = 0
    results_list: List[Dict] = []

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        self.text2image_model = self.load_text2image_model(self.text2image_model_name)
        self.p2p_model_name = 'ip2p'
        self.p2p_model = self.load_pix2pix_model(self.p2p_model_name)

    def text2image(self, prompt_list: List[str]) -> List[Image.Image]:
        image_list = []
        for prompt in prompt_list:
            while True:
                try:
                    images = self.prompt2image(prompt)
                    break
                except Exception as e:
                    print(e)
            image_list.append(images)
        return image_list

    def edit_images(self, image_prompt_list_org: List[str], editing_instructions_list: List[str], batch_size=32) -> Tuple[List[List[Image.Image]], List[str]]:
        image_list = []
        for prompt in image_prompt_list_org:
            image_list.append(self.prompt2image(prompt, images_per_prompt=1)[0])
        image_list = [item for item in image_list if item is not None]
        editing_instructions_list = [item for item, condition in zip(editing_instructions_list, image_list) if condition is not None]
        image_prompt_list_org = [item for item, condition in zip(image_prompt_list_org, image_list) if condition is not None]

        edited_images = self.p2p_model(editing_instructions_list, image_list).images
        all_images = []
        all_prompt = []
        for i in range(len(image_prompt_list_org)*2):
            if i % 2 == 0:
                all_prompt.append(image_prompt_list_org[i//2])
                all_images.append([image_list[i//2]])
            else:
                all_prompt.append(editing_instructions_list[i//2])
                all_images.append([edited_images[i//2]])
        return all_images, all_prompt

    def save_experiment_log(self, activation_list: List[float], image_list: List[str], image_titles: List[str], image_textual_information: List[str] = None):
        output = [{"type": "text", "text": 'Neuron activations:\n'}]
        for ind, act in enumerate(activation_list):
            output.append({"type": "text", "text": f'"{image_titles[ind]}", activation: {act}\nimage: \n'})
            output.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_list[ind]}})
            self.results_list.append({image_titles[ind]: {"activation": act, "image": image_list[ind]}})
        if (self.activation_threshold != 0) and (max(activation_list) < self.activation_threshold):
            output.append({"type": "text", "text": f"\nMax activation is smaller than {(self.activation_threshold * 100).round()/100}, please continue with the experiments.\n"})
        if image_textual_information is not None:
            if isinstance(image_textual_information, list):
                for text in image_textual_information:
                    output.append({"type": "text", "text": text})
            else:
                output.append({"type": "text", "text": image_textual_information})
        self.update_experiment_log(role='user', content=output)

    def update_experiment_log(self, role, content=None, type=None, type_content=None):
        openai_role = {'execution': 'user', 'maia': 'assistant', 'user': 'user', 'system': 'system'}
        if type is None:
            self.experiment_log.append({'role': openai_role[role], 'content': content})
        elif content is None:
            if type == 'text':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type": type, "text": type_content}]})
            if type == 'image_url':
                self.experiment_log.append({'role': openai_role[role], 'content': [{"type": type, "image_url": type_content}]})

    def dataset_exemplars(self, system):
        model_name = system.current_model
        layer = system.current_layer
        neuron_num = system.current_neuron
        image_list = self.dataset_exemplars.exemplars[model_name][layer][neuron_num]
        activation_list = self.dataset_exemplars.activations[model_name][layer][neuron_num]
        self.activation_threshold = sum(activation_list) / len(activation_list)
        activation_list = (activation_list * 100).round() / 100 
        return activation_list, image_list

    def load_pix2pix_model(self, model_name):
        if model_name == "ip2p":
            model_id = "timbrooks/instruct-pix2pix"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe = pipe.to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            return pipe
        else:
            raise ValueError("Unrecognized pix2pix model name")

    def load_text2image_model(self, model_name):
        if model_name == "sd":
            model_id = "runwayml/stable-diffusion-v1-5"
            sdpipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            sdpipe = sdpipe.to(self.device)
            return sdpipe
        elif model_name == "sdxl-turbo":
            model_id = "stabilityai/sdxl-turbo"
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
            pipe = pipe.to(self.device)
            return pipe
        elif model_name == "dalle":
            return None
        else:
            raise ValueError("Unrecognized text to image model name")

    def prompt2image(self, prompt, images_per_prompt=None):
        if images_per_prompt is None:
            images_per_prompt = self.images_per_prompt
        
        if self.text2image_model_name == "sd" or self.text2image_model_name == "sdxl-turbo":
            prompts = [prompt] * images_per_prompt if images_per_prompt > 1 else prompt
            images = self.text2image_model(prompt=prompts, num_inference_steps=4 if self.text2image_model_name == "sdxl-turbo" else 50, guidance_scale=0.0 if self.text2image_model_name == "sdxl-turbo" else 7.5).images
        elif self.text2image_model_name == "dalle":
            if images_per_prompt > 1:
                raise ValueError("Cannot use DALLE with 'images_per_prompt' > 1 due to rate limits")
            images = []
            try:
                response = openai.Image.create(prompt=prompt, n=1, size="256x256")
                image_url = response["data"][0]["url"]
                response = requests.get(image_url)
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                images.append(image)
            except Exception as e:
                print(e)
                images.append(None)
        else:
            raise ValueError("Unrecognized text to image model name")
        return images

    def summarize_images(self, image_list: List[str]) -> str:
        instructions = "What do all the unmasked regions of these images have in common? There might be more than one common concept, or a few groups of images with different common concepts each. In these cases return all of the concepts. Return your description in the following format: [COMMON]: <your description>."
        history = [{'role': 'system', 'content': 'You are a helpful assistant'}]
        user_content = [{"type": "text", "text": instructions}]
        for image in image_list:
            user_content.append({"type": "image_url", "image_url": "data:image/jpeg;base64," + image})
        history.append({'role': 'user', 'content': user_content})
        description = ask_agent('gpt-4-vision-preview', history)
        if isinstance(description, Exception):
            return str(description)
        return description

    def describe_images(self, image_list: List[str], image_title: List[str]) -> str:
        description_list = ''
        instructions = "Do not describe the full image. Please describe ONLY the unmasked regions in this image (e.g. the regions that are not darkened). Be as concise as possible. Return your description in the following format: [highlighted regions]: <your concise description>"
        time.sleep(60)
        for ind, image in enumerate(image_list):
            history = [{'role': 'system', 'content': 'You are a helpful assistant'}, {'role': 'user', 'content': [{"type": "text", "text": instructions}, {"type": "image_url", "image_url": "data:image/jpeg;base64," + image}]}]
            description = ask_agent('gpt-4-vision-preview', history)
            if isinstance(description, Exception):
                return description_list
            description = description.split("[highlighted regions]:")[-1]
            description = " ".join([f'"{image_title[ind]}", highlighted regions:', description])
            description_list += description + '\n'
        return description_list

    def generate_html(self, name="experiment.html"):
        html_string = f'''<html>
        <head>
        <title>Experiment Log</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        </head> 
        <body>
        <h1>{self.path2save}</h1>'''

        for entry in self.experiment_log[2:]:      
            if entry['role'] == 'assistant':
                html_string += f"<h2>MAIA</h2>"  
                html_string += f"<pre>{entry['content'][0]['text']}</pre><br>"
            else:
                html_string += f"<h2>Experiment Execution</h2>"  
                for content_entry in entry['content']:
                    if "image_url" in content_entry["type"]:
                        html_string += f'''<img src="{content_entry['image_url']['url']}"/>'''  
                    elif "text" in content_entry["type"]:
                        html_string += f"<pre>{content_entry['text']}</pre>"
        html_string += '</body></html>'

        with open(os.path.join(self.path2save, name), "w") as file_html:
            file_html.write(html_string)

def generate_masked_image(image,mask,path2save,threshold):
    #Generates a masked image highlighting high activation areas.
    vis = ImageVisualizer(224, image_size=224, source='imagenet')
    masked_tensor = vis.pytorch_masked_image(image, activations=mask, unit=None, level=threshold, outside_bright=0.25) #percent_level=0.95)
    masked_image = Image.fromarray(masked_tensor.permute(1, 2, 0).byte().cpu().numpy())
    buffer = BytesIO()
    masked_image.save(buffer, format="PNG")
    buffer.seek(0)
    masked_image = base64.b64encode(buffer.read()).decode('ascii')
    return(masked_image)

def image2str(image,path2save):
    #Converts an image to a Base64 encoded string.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('ascii')
    return(image)

def str2image(image_str):
    #Converts a Base64 encoded string to an image.
    img_bytes = base64.b64decode(image_str)
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    return img

class DatasetExemplars(BaseModel):
    path2exemplars: str
    n_exemplars: int
    path2save: str
    model_config: Dict[str, Dict[str, List[int]]]
    im_size: int = 224

    exemplars: Dict[str, Dict[str, Dict[int, List[str]]]] = {}
    activations: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    thresholds: Dict[str, Dict[str, Dict[int, float]]] = {}

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_data_structures()
        self._process_all_models()

    def _initialize_data_structures(self):
        self.exemplars = defaultdict(lambda: defaultdict(dict))
        self.activations = defaultdict(lambda: defaultdict(dict))
        self.thresholds = defaultdict(lambda: defaultdict(dict))

    def _process_all_models(self):
        for model_name, layers in self.model_config.items():
            for layer, units in layers.items():
                exemplars, activations, thresholds = self.net_dissect(model_name, layer, units)
                self.exemplars[model_name][layer] = exemplars
                self.activations[model_name][layer] = activations
                self.thresholds[model_name][layer] = thresholds

    def net_dissect(self, model_name: str, layer: str, units: List[int]) -> tuple:
        exp_path = f'{self.path2exemplars}/{model_name}/imagenet/{layer}'
        activations = np.loadtxt(f'{exp_path}/activations.csv', delimiter=',')
        thresholds = np.loadtxt(f'{exp_path}/thresholds.csv', delimiter=',')
        image_array = np.load(f'{exp_path}/images.npy')
        mask_array = np.load(f'{exp_path}/masks.npy')

        all_images = []
        filtered_activations = []
        filtered_thresholds = []

        for unit in units:
            curr_image_list = []
            for exemplar_idx in range(min(activations.shape[1], self.n_exemplars)):
                save_path = os.path.join(self.path2save, 'dataset_exemplars', model_name, layer, str(unit), 'netdisect_exemplars')
                file_path = os.path.join(save_path, f'{exemplar_idx}.png')

                if os.path.exists(file_path):
                    with open(file_path, "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                else:
                    curr_mask = np.repeat(mask_array[unit, exemplar_idx], 3, axis=0)
                    curr_image = image_array[unit, exemplar_idx]
                    inside = np.array(curr_mask > 0)
                    outside = np.array(curr_mask == 0)
                    masked_image = curr_image * inside + 0.25 * curr_image * outside
                    masked_image = Image.fromarray(np.transpose(masked_image, (1, 2, 0)).astype(np.uint8))
                    masked_image = masked_image.resize([self.im_size, self.im_size], Image.Resampling.LANCZOS)
                    os.makedirs(save_path, exist_ok=True)
                    masked_image.save(file_path, format='PNG')
                    with open(file_path, "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')

                curr_image_list.append(masked_image)
            
            all_images.append(curr_image_list)
            filtered_activations.append(activations[unit, :self.n_exemplars])
            filtered_thresholds.append(thresholds[unit])

        return all_images, np.array(filtered_activations), np.array(filtered_thresholds)

    def get_exemplars(self, model_name: str, layer: str, unit: int) -> List[str]:
        return self.exemplars[model_name][layer][unit]

    def get_activations(self, model_name: str, layer: str, unit: int) -> List[float]:
        return self.activations[model_name][layer][unit]

    def get_threshold(self, model_name: str, layer: str, unit: int) -> float:
        return self.thresholds[model_name][layer][unit]