# Standard library imports
import math
import os
from typing import Any, Dict, List, Tuple, Union

import clip
import numpy as np

# Third-party imports
import openai
import torch
import torch.nn.functional as F
import torchvision.models as models
from baukit import Trace
from PIL import Image
from torchvision import transforms

from synthetic_neurons_dataset.synthetic_neurons import SAMNeuron

# Local imports
from utils.agents.factory import create_agent
from utils.api_utils import (
    format_api_content,
    generate_masked_image,
    image2str,
    is_base64,
    str2image,
)
from utils.DatasetExemplars import DatasetExemplars


class System:
    """
    A Python class containing the vision model and the specific neuron to interact with.

    Attributes
    ----------
    neuron_num : int
        The serial number of the neuron.
    layer : string
        The name of the layer where the neuron is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    neuron : callable
        A lambda function to compute neuron activation and activation map per input image.
        Use this function to test the neuron activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_neuron(image_list: List[torch.Tensor]) -> Tuple[List[int], List[str]]
        Returns the neuron activation for each image in the input image_list as well as the original image (encoded into a Base64 string).
    """

    def __init__(
        self, neuron_num: int, layer: str, model_name: str, device: str, thresholds=None
    ):
        """
        Initializes a neuron object by specifying its number and layer location and the vision model that the neuron belongs to.
        Parameters
        -------
        neuron_num : int
            The serial number of the neuron.
        layer : str
            The name of the layer that the neuron is located at.
        model_name : str
            The name of the vision model that the neuron is part of.
        device : str
            The computational device ('cpu' or 'cuda').
        """
        self.neuron_num = neuron_num
        self.layer = layer
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.preprocess = None
        if "dino" in model_name or "resnet" in model_name:
            self.preprocess = self._preprocess_imagenet
        self.model = self._load_model(model_name)  # if clip, the define self.preprocess
        if thresholds is not None:
            self.threshold = thresholds[self.layer][self.neuron_num]
        else:
            self.threshold = 0

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """
        Gets the model name and returns the vision model from pythorch library.
        Parameters
        ----------
        model_name : str
            The name of the model to load.

        Returns
        -------
        nn.Module
            The loaded PyTorch vision model.

        Examples
        --------
        >>> # load "resnet152"
        >>> def execute_command(model_name) -> nn.Module:
        >>>   model = load_model(model_name: str)
        >>>   return model
        """
        if model_name == "resnet152":
            resnet152 = models.resnet152(weights="IMAGENET1K_V1").to(self.device)
            model = resnet152.eval()
        elif model_name == "dino_vits8":
            model = (
                torch.hub.load("facebookresearch/dino:main", "dino_vits8")
                .to(self.device)
                .eval()
            )
        elif model_name == "clip-RN50":
            name = "RN50"
            full_model, preprocess = clip.load(
                name, download_root="/data/scratch/yusepp/.cache/clip"
            )
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        elif model_name == "clip-ViT-B32":
            name = "ViT-B/32"
            full_model, preprocess = clip.load(name)
            model = full_model.visual.to(self.device).eval()
            self.preprocess = preprocess
        return model

    def call_neuron(self, image_list: List[str]) -> Tuple[List[int], List[str]]:
        """
        The function returns the neuron’s maximum activation value (in int format) for each of the images in the list as well as the original image (encoded into a Base64 string).

        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image

        Returns
        -------
        Tuple[List[int], List[str]]
            For each image in image_list returns the activation value of the neuron on that image, and the original image encoded into a Base64 string.


        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> prompt = ['a dog standing on the grass']
        >>> image = tools.text2image(prompt)
        >>> activation_list, image_list = system.call_neuron(image)
        >>> for activation, image in zip(activation_list, image_list):
        >>>     tools.display(image, f"Activation: {activation}")
        >>>
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass" and maintain robustness to noise
        >>> prompts = ['a dog standing on the grass'] * 5
        >>> images = tools.text2image(prompts)
        >>> activation_list, image_list = system.call_neuron(images)
        >>> tools.display(image_list[0], f'Activation: {statistics.mean(activation_list)}')
        >>>
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass" and the neuron activation value for the same image but with a lion instead of a dog
        >>> prompt = ['a dog standing on the grass']
        >>> edits = ['replace the dog with a lion']
        >>> all_images, all_prompts = tools.edit_images(prompt, edits)
        >>> activation_list, image_list = system.call_neuron(all_images)
        >>> for activation, image in zip(activation_list, image_list):
        >>>     tools.display(image, f"Activation: {activation}")
        """
        activation_list = []
        masked_images_list = []
        for image in image_list:
            if image == None:  # for dalle
                activation_list.append(None)
                masked_images_list.append(None)
            else:
                image = str2image(image)
                if self.layer == "last":
                    tensor = self._preprocess_images(image)
                    acts, image_class = self._calc_class(tensor)
                    activation_list.append(torch.round(acts[ind] * 100).item() / 100)
                    masked_images_list.append(image2str(image[0]))
                else:
                    image = self._preprocess_images(image)
                    acts, masks = self._calc_activations(image)
                    ind = torch.argmax(acts).item()
                    masked_image = generate_masked_image(
                        image[ind], masks[ind], "./temp.png", self.threshold
                    )
                    activation_list.append(torch.round(acts[ind] * 100).item() / 100)
                    masked_images_list.append(masked_image)
        return activation_list, masked_images_list

    @staticmethod
    def _spatialize_vit_mlp(hiddens: torch.Tensor) -> torch.Tensor:
        """Make ViT MLP activations look like convolutional activations.

        Each activation corresponds to an image patch, so we can arrange them
        spatially. This allows us to use all the same dissection tools we
        used for CNNs.

        Args:
            hiddens: The hidden activations. Should have shape
                (batch_size, n_patches, n_units).

        Returns:
            Spatially arranged activations, with shape
                (batch_size, n_units, sqrt(n_patches - 1), sqrt(n_patches - 1)).
        """
        batch_size, n_patches, n_units = hiddens.shape

        # Exclude CLS token.
        hiddens = hiddens[:, 1:]
        n_patches -= 1

        # Compute spatial size.
        size = math.isqrt(n_patches)
        assert size**2 == n_patches

        # Finally, reshape.
        return hiddens.permute(0, 2, 1).reshape(batch_size, n_units, size, size)

    def _calc_activations(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """ "
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).

        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.

        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask

        Examples
        --------
        >>> # load neuron 62, layer4 of resnet152
        >>> def execute_command(model_name) -> callable:
        >>>   model = load_model(model_name: str)
        >>>   neuron = load_neuron(neuron_num=62, layer='layer4', model=model)
        >>>   return neuron
        """
        with Trace(self.model, self.layer) as ret:
            _ = self.model(image)
            hiddens = ret.output

        if "dino" in self.model_name:
            hiddens = self._spatialize_vit_mlp(hiddens)

        batch_size, channels, *_ = hiddens.shape
        activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
        neuron_activation_map = hiddens[:, self.neuron_num, :, :]
        return (pooled[:, self.neuron_num], neuron_activation_map)

    def _calc_class(self, image: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """ "
        Returns the neuron activation for the input image, as well as the activation map of the neuron over the image
        that highlights the regions of the image where the activations are higher (encoded into a Base64 string).

        Parameters
        ----------
        image : torch.Tensor
            The input image in PIL format.

        Returns
        -------
        Tuple[int, torch.Tensor]
            Returns the maximum activation value of the neuron on the input image and a mask

        Examples
        --------
        >>> # load neuron 62, layer4 of resnet152
        >>> def execute_command(model_name) -> callable:
        >>>   model = load_model(model_name: str)
        >>>   neuron = load_neuron(neuron_num=62, layer='layer4', model=model)
        >>>   return neuron
        """
        logits = self.model(image)
        prob = F.softmax(logits, dim=1)
        image_calss = torch.argmax(logits[0])
        activation = prob[0][image_calss]
        return activation.item(), image

    def _preprocess_imagenet(self, image, normalize=True, im_size=224):
        if normalize:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                ]
            )
        return preprocess(image)

    def _preprocess_images(self, images):
        image_list = []
        if type(images) == list:
            for image in images:
                image_list.append(self.preprocess(image).to(self.device))
            batch_tensor = torch.stack(image_list)
            return batch_tensor
        else:
            return self.preprocess(images).unsqueeze(0).to(self.device)


class Synthetic_System:
    """
    A Python class containing the vision model and the specific neuron to interact with.

    Attributes
    ----------
    neuron_num : int
        The serial number of the neuron.
    layer : string
        The name of the layer where the neuron is located.
    model_name : string
        The name of the vision model.
    model : nn.Module
        The loaded PyTorch model.
    neuron : callable
        A lambda function to compute neuron activation and activation map per input image.
        Use this function to test the neuron activation for a specific image.
    device : torch.device
        The device (CPU/GPU) used for computations.

    Methods
    -------
    load_model(model_name: str)->nn.Module
        Gets the model name and returns the vision model from PyTorch library.
    call_neuron(image_list: List[torch.Tensor])->Tuple[List[int], List[str]]
        returns the neuron activation for each image in the input image_list as well as the activation map
        of the neuron over that image, that highlights the regions of the image where the activations
        are higher (encoded into a Base64 string).
    """

    def __init__(
        self, neuron_num: int, neuron_labels: str, neuron_mode: str, device: str
    ):
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.neuron_num = neuron_num
        self.neuron_labels = neuron_labels
        self.neuron = SAMNeuron(neuron_labels, neuron_mode, device=self.device)
        self.threshold = 0
        self.layer = neuron_mode

    def call_neuron(
        self, image_list: List[torch.Tensor]
    ) -> Tuple[List[int], List[str]]:
        """
        The function returns the neuron’s maximum activation value (in int format) over each of the images in the list as well as the activation map of the neuron over each of the images that highlights the regions of the image where the activations are higher (encoded into a Base64 string).

        Parameters
        ----------
        image_list : List[torch.Tensor]
            The input image

        Returns
        -------
        Tuple[List[int], List[str]]
            For each image in image_list returns the maximum activation value of the neuron on that image, and a masked images,
            with the region of the image that caused the high activation values highlighted (and the rest of the image is darkened). Each image is encoded into a Base64 string.


        Examples
        --------
        >>> # test the activation value of the neuron for the prompt "a dog standing on the grass"
        >>> def execute_command(system, prompt_list) -> Tuple[int, str]:
        >>>     prompt = ["a dog standing on the grass"]
        >>>     image = text2image(prompt)
        >>>     activation_list, activation_map_list = system.call_neuron(image)
        >>>     return activation_list, activation_map_list
        >>> # test the activation value of the neuron for the prompt “a fox and a rabbit watch a movie under a starry night sky” “a fox and a bear watch a movie under a starry night sky” “a fox and a rabbit watch a movie at sunrise”
        >>> def execute_command(system.neuron, prompt_list) -> Tuple[int, str]:
        >>>     prompt_list = [[“a fox and a rabbit watch a movie under a starry night sky”, “a fox and a bear watch a movie under a starry night sky”,“a fox and a rabbit watch a movie at sunrise”]]
        >>>     images = text2image(prompt_list)
        >>>     activation_list, activation_map_list = system.call_neuron(images)
        >>>     return activation_list, activation_map_list
        """
        activation_list = []
        masked_images_list = []
        for image in image_list:
            if image == None:  # for dalle
                activation_list.append(None)
                masked_images_list.append(None)
            else:
                acts, _, _, masks = self.neuron.calc_activations(image)
                ind = np.argmax(acts)
                masked_image = image2str(masks[ind])
                activation_list.append(acts[ind])
                masked_images_list.append(masked_image)
        return activation_list, masked_images_list


class Tools:
    """
    A Python class containing tools to interact with the units implemented in the system class,
    in order to run experiments on it.

    Attributes
    ----------
    text2image_model : any
        The loaded text-to-image model.
    img2img_model: any
        The loaded image-to-image model.
    path2save : str
        Path for saving output images.
    threshold : any
        Activation threshold for neuron analysis.
    device : torch.device
        The device (CPU/GPU) used for computations.
    experiment_log: str
        A log of all the experiments, including the code and the output from the neuron
        analysis.
    exemplars : Dict
        A dictionary containing the exemplar images for each unit.
    exemplars_activations : Dict
        A dictionary containing the activations for each exemplar image.
    exemplars_thresholds : Dict
        A dictionary containing the threshold values for each unit.
    results_list : List
        A list of the results from the neuron analysis.


    Methods
    -------
    dataset_exemplars(self, unit_ids: List[int], system: System)->List[List[Tuple[float, str]]]:
        Retrieves the activations and exemplar images for a list of units.
    edit_images(self, image_prompts : List[Image.Image], editing_prompts : List[str]):
        Generate images from a list of prompts, then edits each image with the
        corresponding editing prompt.
    text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.
    summarize_images(self, image_list: List[str]) -> str:
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.
    sampler(act: any, imgs: List[any], mask: any, prompt: str, method: str = 'max') -> Tuple[List[int], List[str]]
        Processes images based on neuron activations.
    describe_images(self, image_list: List[str], image_title:List[str]) -> str:
        Generates descriptions for a list of images, focusing specifically on highlighted regions.
    display(self, *args: Union[str, Image.Image]):
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.

    """

    def __init__(
        self,
        path2save: str,
        device: str,
        DatasetExemplars: DatasetExemplars = None,
        image2text_model_name="gpt-4o",
        text2image_model: Any = None,
        img2img_model: Any = None,
    ):
        """
        Initializes the Tools object.

        Parameters
        ----------
        path2save : str
            Path for saving output images.
        device : str
            The computational device ('cpu' or 'cuda').
        DatasetExemplars : object
            an object from the class DatasetExemplars
        text2image_model : any
            The loaded text-to-image model.
        img2img_model: any
            The loaded image-to-image model.
        """
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.image2text_model_name = image2text_model_name
        self.text2image_model = text2image_model
        self.img2img_model = img2img_model
        self.experiment_log = []
        self.im_size = 224
        if DatasetExemplars is not None:
            self.exemplars = DatasetExemplars.exemplars
            self.exemplars_activations = DatasetExemplars.activations
            self.exempalrs_thresholds = DatasetExemplars.thresholds
        self.activation_threshold = 0
        self.results_list = []

    def dataset_exemplars(self, system: System) -> List[List[Tuple[float, str]]]:
        """
        This method finds images from the ImageNet dataset that produce the highest activation values for a specific neuron.
        It returns both the activation values and the corresponding exemplar images that were used to generate these activations.
        This experiment is performed on real images and will provide a good approximation of the neuron behavior.

        Parameters
        ----------
        system : System
            The system representing the specific neuron and layer within the neural network.
            The system should have 'layer' and 'neuron_num' attributes, so the dataset_exemplars function
            can return the exemplar activations and images for that specific neuron.

        Returns
        -------
        List
            For each exemplar image, stores a tuple containing two elements:
            - The first element is the activation value for the specified neuron.
            - The second element is the exemplar images (as Base64 encoded strings) corresponding to the activation.

        Example
        -------
        >>> exemplar_data = tools.dataset_exemplars(system)
        >>> for activation, image in exemplar_data:
        >>>    tools.display(image, f"Activation: {activation}")
        """
        image_list = self.exemplars[system.layer][system.neuron_num]
        activation_list = self.exemplars_activations[system.layer][system.neuron_num]
        self.activation_threshold = sum(activation_list) / len(activation_list)
        activation_list = (activation_list * 100).round() / 100
        return list(zip(activation_list, image_list))

    def edit_images(
        self, base_images: List[str], editing_prompts: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Generates or uses provided base images, then edits each base image with a
        corresponding editing prompt. Accepts either text prompts or Base64
        encoded strings as sources for the base images.

        The function returns a list containing lists of images (original and edited,
        interleaved) in Base64 encoded string format, and a list of the relevant
        prompts (original source string and editing prompt, interleaved).

        Parameters
        ----------
        base_images : List[str]
            A list of images as Base64 encoded strings. These images are to be
            edited by the prompts in editing_prompts.
        editing_prompts : List[str]
            A list of instructions for how to edit the base images derived from
            `base_images`. Must be the same length as `base_images`.

        Returns
        -------
        Tuple[List[str], List[str]]
            - all_images: A list where elements alternate between:
                - A Base64 string for the original image from a source.
                - A Base64 string for the edited image from that source.
              Example: [orig1_img1, edit1_img1, orig2_img1, edit2_img1, ...]
            - all_prompts: A list where elements alternate between:
                - The original source string (text prompt or Base64) used.
                - The editing prompt used.
              Example: [source1, edit1, source2, edit2, ...]
            The order in `all_images` corresponds to the order in `all_prompts`.

        Raises
        ------
        ValueError
            If the lengths of `base_images` and `editing_prompts` are not equal.

        Examples
        --------
        >>> # test the confidence score of the classifier for the prompt "a dog standing on the grass"
        >>> # for the same image but with different actions instead of "standing":
        >>> prompts = ['a landscape with a tree and a river'] * 3
        >>> original_images = tools.text2image(prompts)
        >>> edits = ['make it autumn', 'make it spring', 'make it winter']
        >>> all_images, all_prompts = tools.edit_images(original_images, edits)
        >>> score_list, image_list = system.call_classifier(all_images)
        >>> for score, image, prompt in zip(score_list, image_list, all_prompts):
        >>>     tools.display(image, f"Prompt: {prompt}\nConfidence Score: {score}")
        >>>
        >>> # test the confidence score of the classifier on the highest scoring dataset exemplar
        >>> # under different conditions
        >>> exemplar_data = tools.dataset_exemplars(system)
        >>> highest_scoring_exemplar = exemplar_data[0][1]
        >>> edits = ['make it night', 'make it daytime', 'make it snowing']
        >>> all_images, all_prompts = tools.edit_images(
        ...     [highest_scoring_exemplar] * len(edits), edits
        ... )
        >>> score_list, image_list = system.call_classifier(all_images)
        >>> for score, image, prompt in zip(score_list, image_list, all_prompts):
        >>>     tools.display(image, f"Prompt: {prompt}\nConfidence Score: {score}")
        """
        if len(base_images) != len(editing_prompts):
            raise ValueError("Length of base_images and editing_prompts must be equal.")
        if isinstance(editing_prompts, str):
            editing_prompts = [editing_prompts]

        base_imgs_obj = [str2image(img_b64) for img_b64 in base_images]
        edited_images_b64_lists = self.img2img_model(editing_prompts, base_imgs_obj)
        edited_images_b64_lists = [image2str(img) for img in edited_images_b64_lists]

        # Interleave results
        all_images = []
        all_prompts = []
        for i in range(len(base_images)):
            all_images.append(base_images[i])
            all_prompts.append("Original Image")

            all_images.append(edited_images_b64_lists[i])
            all_prompts.append(f"Editing Prompt: {editing_prompts[i]}")

        return all_images, all_prompts

    def text2image(self, prompt_list: List[str]) -> List[torch.Tensor]:
        """Gets a list of text prompt as an input, generates an image for each prompt in the list using a text to image model.
        The function returns a list of images.

        Parameters
        ----------
        prompt_list : List[str]
            A list of text prompts for image generation.

        Returns
        -------
        List[Image.Image]
            A list of images, corresponding to each of the input prompts.


        Examples
        --------
        >>> # Generate images from a list of prompts
        >>>     prompt_list = [“a toilet on mars”,
        >>>                     “a toilet on venus”,
        >>>                     “a toilet on pluto”]
        >>>     images = tools.text2image(prompt_list)
        >>>     tools.display(*images)
        """
        images = self.text2image_model(prompt_list)
        return [image2str(img) for img in images]

    def summarize_images(self, image_list: List[str]) -> str:
        """
        Gets a list of images and describes what is common to all of them, focusing specifically on unmasked regions.


        Parameters
        ----------
        image_list : list
            A list of images in Base64 encoded string format.

        Returns
        -------
        str
            A string with a descriptions of what is common to all the images.

        Example
        -------
        >>> # Summarize a unit's dataset exemplars
        >>> _, exemplars = tools.dataset_exemplars([0], system)[0]  # Get exemplars for unit 0
        >>> summarization = tools.summarize_images(image_list)
        >>> tools.display('Unit 0 summarization: ', summarization)
        >>>
        >>> # Summarize what's common amongst two sets of exemplars
        >>> exemplars_data = tools.dataset_exemplars([0, 1], system)
        >>> all_exemplars = []
        >>> for _, exemplars in exemplars_data:
        >>>     all_exemplars += exemplars
        >>> summarization = tools.summarize_images(all_exemplars)
        >>> tools.display('All exemplars summarization: ', summarization)
        First, list the common features that you see in the regions such as:

        Non-semantic Concepts (Shape, texture, color, etc.): ...
        Semantic Concepts (Objects, animals, people, scenes, etc): ...
        """
        image_list = self._description_helper(image_list)
        instructions = "What do all the unmasked regions of these images have in common? There might be more then one common concept, or a few groups of images with different common concept each. In these cases return all of the concepts.. Return your description in the following format: [COMMON]: <your description>."
        history = [
            {
                "role": "system",
                "content": "You are a helpful assistant who views/compares images.",
            }
        ]
        user_content = [{"type": "text", "text": instructions}]
        for ind, image in enumerate(image_list):
            user_content.append(format_api_content("image_url", image))
        history.append({"role": "user", "content": user_content})

        agent = create_agent(
            model=self.image2text_model_name, max_attempts=5, max_output_tokens=4096
        )

        description = agent.ask(history)
        if isinstance(description, Exception):
            return description
        return description

    def sampler(self, act, imgs, mask, prompt, threshold, method="max"):
        if method == "max":
            max_ind = torch.argmax(act).item()
            masked_image_max = self.generate_masked_image(
                image=imgs[max_ind],
                mask=mask[max_ind],
                path2save=f"{self.path2save}/{prompt}_masked_max.png",
                threshold=threshold,
            )
            acts = max(act).item()
            ims = masked_image_max
        elif method == "min_max":
            max_ind = torch.argmax(act).item()
            min_ind = torch.argmin(act).item()
            masked_image_max = self.generate_masked_image(
                image=imgs[max_ind],
                mask=mask[max_ind],
                path2save=f"{self.path2save}/{prompt}_masked_max.png",
                threshold=threshold,
            )
            masked_image_min = self.generate_masked_image(
                image=imgs[min_ind],
                mask=mask[min_ind],
                path2save=f"{self.path2save}/{prompt}_masked_min.png",
                threshold=threshold,
            )
            acts = []
            ims = []
            acts.append(max(act).item())
            acts.append(min(act).item())
            ims.append(masked_image_max)
            ims.append(masked_image_min)
        return acts, ims

    def describe_images(self, image_list: List[str], image_title: List[str]) -> str:
        """
        Provides impartial description of the highlighted image regions within an image.
        Generates textual descriptions for a list of images, focusing specifically on highlighted regions.
        This function translates the visual content of the highlighted region in the image to a text description.
        The function operates independently of the current hypothesis list and thus offers an impartial description of the visual content.
        It iterates through a list of images, requesting a description for the
        highlighted (unmasked) regions in each synthetic image. The final descriptions are concatenated
        and returned as a single string, with each description associated with the corresponding
        image title.

        Parameters
        ----------
        image_list : List[str]
            A list of images in Base64 encoded string format.
        image_title : List[str]
            A list of titles for each image in the image_list.

        Returns
        -------
        str
            A concatenated string of descriptions for each image, where each description
            is associated with the image's title and focuses on the highlighted regions
            in the image.

        Example
        -------
        >>> prompt_list = [“a man with two teeth”,
                            “a man with fangs”,
                            “a man with all molars, like a horse”]
        >>> images = tools.text2image(prompt_list)
        >>> activation_list, masked_images = system.units(images, [0])[0]
        >>> descriptions = tools.describe_images(masked_images, prompt_list)
        >>> tools.display(*descriptions)
        """
        description_list = ""
        instructions = "Do not describe the full image. Please describe ONLY the unmasked regions in this image (e.g. the regions that are not darkened). Be as concise as possible. Return your description in the following format: [HIGHLIGHTED REGIONS]: <your concise description>"
        image_list = self._description_helper(image_list)
        for ind, image in enumerate(image_list):
            history = [
                {"role": "system", "content": "you are an helpful assistant"},
                {
                    "role": "user",
                    "content": [
                        format_api_content("text", instructions),
                        format_api_content("image_url", image),
                    ],
                },
            ]
            agent = create_agent(
                model=self.image2text_model_name, max_attempts=5, max_output_tokens=4096
            )
            description = agent.ask(history)
            if isinstance(description, Exception):
                return description_list
            description = description.split("[HIGHLIGHTED REGIONS]:")[-1]
            description = " ".join(
                [f'"{image_title[ind]}", HIGHLIGHTED REGIONS:', description]
            )
            description_list += description + "\n"
        return description_list

    def _description_helper(self, *args: Union[str, Image.Image]):
        """Helper function for display to recursively handle iterable arguments."""
        output = []
        for item in args:
            if isinstance(item, (list, tuple)):
                output.extend(self._description_helper(*item))
            else:
                output.append(item)
        return output

    def display(self, *args: Union[str, Image.Image]):
        """
        Displays a series of images and/or text in the chat, similar to a Jupyter notebook.

        Parameters
        ----------
        *args : Union[str, Image.Image]
            The content to be displayed in the chat. Can be multiple strings or Image objects.

        Notes
        -------
        Displays directly to chat interface.

        Example
        -------
        >>> # Display a single image
        >>> prompt = ['a dog standing on the grass']
        >>> images = tools.text2image(prompt)
        >>> tools.display(*images)
        >>>
        >>> # Display a list of images
        >>> prompt_list = ["A green creature",
        >>>                 "A red creature",
        >>>                 "A blue creature"]
        >>> images = tools.text2image(prompt_list)
        >>> tools.display(*images)
        """
        output = []
        for item in args:
            # Check if tuple or list
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        self.update_experiment_log(role="user", content=output)

    def _display_helper(self, *args: Union[str, Image.Image]):
        """Helper function for display to recursively handle iterable arguments."""
        output = []
        for item in args:
            if isinstance(item, (list, tuple)):
                output.extend(self._display_helper(*item))
            else:
                output.append(self._process_chat_input(item))
        return output

    def update_experiment_log(self, role, content=None, type=None, type_content=None):
        openai_role = {
            "execution": "user",
            "maia": "assistant",
            "user": "user",
            "system": "system",
        }
        if type == None:
            self.experiment_log.append({"role": openai_role[role], "content": content})
        elif content == None:
            if type == "text":
                self.experiment_log.append(
                    {
                        "role": openai_role[role],
                        "content": [{"type": type, "text": type_content}],
                    }
                )
            if type == "image_url":
                self.experiment_log.append(
                    {
                        "role": openai_role[role],
                        "content": [{"type": type, "image_url": type_content}],
                    }
                )  # gemini

    def _process_chat_input(self, content: Union[str, Image.Image]) -> Dict[str, str]:
        """Processes the input content for the chatbot.

        Parameters
        ----------
        content : Union[str, Image.Image]
            The input content to be processed."""

        if is_base64(content):
            return format_api_content("image_url", content)
        elif isinstance(content, Image.Image):
            return format_api_content("image_url", image2str(content))
        else:
            return format_api_content("text", content)

    def _generate_safe_images(self, prompts: List[str], max_attempts: int = 10):
        results = []
        for prompt in prompts:
            safe_image = self._generate_single_safe_image(prompt, max_attempts)
            results.append(safe_image)
        return results

    def _generate_single_safe_image(self, prompt, max_attempts):
        for attempt in range(max_attempts):
            # Generate the image
            result = self.text2image_model(prompt)

            # Check if the image is safe (not NSFW)
            if not result.nsfw_content_detected[0]:
                return result.images[0]  # Return the safe image

            print(
                f"Prompt '{prompt}': Attempt {attempt + 1}: NSFW content detected. Retrying..."
            )

        raise Exception(
            f"Prompt '{prompt}': Failed to generate a safe image after {max_attempts} attempts"
        )

    def generate_html(self, path2save, name="experiment", line_length=100):
        # Generates an HTML file with the experiment log.
        html_string = f"""<html>
        <head>
        <title>Experiment Log</title>
        <!-- Include Prism Core CSS (Choose the theme you prefer) -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <!-- Include Prism Core JavaScript -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <!-- Include the Python language component for Prism -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        </head> 
        <body>
        <h1>{path2save}</h1>"""

        # don't plot system+user prompts (uncomment if you want the html to include the system+user prompts)
        """
        html_string += f"<h2>{self.experiment_log[0]['role']}</h2>"
        html_string += f"<pre><code>{self.experiment_log[0]['content']}</code></pre><br>"
        
        html_string += f"<h2>{self.experiment_log[1]['role']}</h2>"
        html_string += f"<pre>{self.experiment_log[1]['content'][0]}</pre><br>"
        initial_images = ''
        initial_activations = ''
        for cont in self.experiment_log[1]['content'][1:]:
            if isinstance(cont, dict):
                initial_images += f"<img src="data:image/png;base64,{cont['image']}"/>"
            else:
                initial_activations += f"{cont}    "
        html_string +=  initial_images
        html_string += f"<p>Activations:</p>"
        html_string += initial_activations
        """
        for entry in self.experiment_log:
            if entry["role"] == "assistant":
                html_string += "<h2>MAIA</h2>"
                text = entry["content"][0]["text"]
                # Wrap text to line_length
                # text = textwrap.fill(text, line_length)

                html_string += f"<pre>{text}</pre><br>"
                html_string += "<h2>Experiment Execution</h2>"
            else:
                for content_entry in entry["content"]:
                    if "image_url" in content_entry["type"]:
                        html_string += (
                            f"""<img src="{content_entry["image_url"]["url"]}"/>"""
                        )
                    elif "text" in content_entry["type"]:
                        html_string += f"<pre>{content_entry['text']}</pre>"
        html_string += "</body></html>"

        # Save
        file_path = os.path.join(path2save, f"{name}.html")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(html_string)
