import json
import os
import random
import sys
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

sys.path.append("./synthetic_neurons_dataset/Grounded-Segment-Anything/")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
from GroundingDINO import groundingdino as groundingdino
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# segment anything
from segment_anything import SamPredictor, sam_model_registry


class SAMNeuron:
    def __init__(
        self,
        labels,
        mode,
        pathe2sam="./synthetic_neurons_dataset/Grounded-Segment-Anything/",
        device="cpu",
    ):
        config_file = (
            pathe2sam + "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        grounded_checkpoint = pathe2sam + "groundingdino_swint_ogc.pth"
        sam_version = "vit_h"
        sam_checkpoint = pathe2sam + "sam_vit_h_4b8939.pth"
        self.device = device
        self.model = self.load_model(
            config_file,
            grounded_checkpoint,
        )
        self.predictor = SamPredictor(
            sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(self.device)
        )
        self.labels = labels
        self.mode = mode

    def load_image(self, image_pil):
        transform = T.Compose(
            [
                # T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def decode_b64(self, b64_string):
        """
        Decode a base64 string to a PIL image.
        """
        import base64
        from io import BytesIO

        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data))
        return image.convert("RGB")

    def load_model(self, model_config_path, model_checkpoint_path):
        print(model_config_path)
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(
        self,
        model,
        image,
        caption,
        box_threshold,
        text_threshold,
        with_logits=True,
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def show_mask(self, mask, image):
        image_data = (image.astype(np.float32) - image.min()) / (
            image.max() - image.min()
        )
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        dilated_mask = self.dilate(mask_image).reshape(h, w, 1)
        expanded_mask = np.repeat(dilated_mask, 3, axis=2)
        # print(expanded_mask.shape)
        # print(image.shape)
        darkening_factor = 0.25
        darkening_mask = np.ones_like(image_data)
        darkening_mask[expanded_mask == 0] = darkening_factor
        darkened_image = image_data * darkening_mask
        return darkened_image, expanded_mask

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
        ax.text(x0, y0, label)

    def dilate(self, binary_mask, dilation_factor=0.3):
        """
        Dilate each connected component in a binary image mask by a scaling factor.

        :param binary_mask: A binary image mask with values 0 and 1.
        :param dilation_factor: Factor to scale the structuring element for each component.
        :return: Dilated binary image mask.
        """
        # Convert the binary mask to a format suitable for OpenCV operations
        mask = np.uint8(binary_mask) * 255
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask)
        # Create an empty image for the dilated mask
        dilated_mask = np.zeros_like(mask)

        for label in range(1, num_labels):  # Label 0 is the background
            # Extract the individual component
            component_mask = (labels == label).astype(np.uint8)

            # Calculate the size of the structuring element based on the component size and dilation factor
            element_size = 5  # int(np.sqrt(component_area) * dilation_factor)

            # Create the structuring element for dilation
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (element_size, element_size)
            )
            # Apply dilation to the individual component
            dilated_component = cv2.dilate(component_mask * 255, kernel, iterations=9)

            # Add the dilated component to the final mask
            dilated_mask = cv2.bitwise_or(dilated_mask, dilated_component)
        # Convert back to binary format (0s and 1s)
        dilated_mask_binary = dilated_mask / 255
        return dilated_mask_binary

    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "mask.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        json_data = [{"value": value, "label": "background"}]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split("(")
            logit = logit[:-1]  # the last is ')'
            json_data.append(
                {
                    "value": value,
                    "label": name,
                    "logit": float(logit),
                    "box": box.numpy().tolist(),
                }
            )
        with open(os.path.join(output_dir, "mask.json"), "w") as f:
            json.dump(json_data, f)

    def calc_activations(
        self, images
    ):  # calculate mask and activations for an input image

        # set parameters
        box_threshold = 0.25
        text_threshold = 0.3
        text_prompt_list = self.labels
        mask_per_label = []
        logit_per_label = []
        check_lables = []
        logit = 0

        # initialize output lists
        activations = []
        masks_out = []
        images_out = []
        masked_images = []

        # loop over images
        if isinstance(images, str):
            images = [self.decode_b64(images)]

        for image_pil in images:
            mask = None
            # loop over text prompts
            for text_prompt in text_prompt_list:
                image_pil = image_pil.resize((224, 224), Image.LANCZOS)
                image = self.load_image(image_pil)

                # run grounding dino model
                boxes_filt, pred_phrases = self.get_grounding_output(
                    self.model, image, text_prompt, box_threshold, text_threshold
                )

                image = np.array(image_pil)
                self.predictor.set_image(image)

                # get masks
                H, W = 224, 224
                if boxes_filt.shape[0] != 0:
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = self.predictor.transform.apply_boxes_torch(
                        boxes_filt, image.shape[:2]
                    ).to(self.device)
                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(self.device),
                        multimask_output=False,
                    )
                else:
                    masks = torch.empty(1, 1, H, W)
                    masks[:, :, :, :] = False

                if mask is None:
                    mask = masks[0].cpu().numpy()

                # draw output image
                check_lables.append(pred_phrases != [])
                cnt = 0
                if pred_phrases != []:  # if the synthetic neuron is activated
                    logit = 0.0
                    for i, (x, y) in enumerate(zip(masks, pred_phrases)):
                        mask = mask + x.cpu().numpy()
                        _, l = y.split("(")
                        logit += float(l[:-1])
                        cnt += 1
                else:  # if the synthetic neuron is not activated
                    mask = masks[0].cpu().numpy()
                    logit += 0
                    cnt += 1

                mask_per_label.append(mask)
                logit_per_label.append(logit / cnt)

            # combine masks and logit according to the synthetic neuron mode
            if self.mode == "mono":
                mask = sum(mask_per_label)
                final_logit = sum(logit_per_label) / len(logit_per_label)
            elif self.mode == "or":
                final_logit = max(logit_per_label)
                mask = sum(mask_per_label)
            elif self.mode == "and":
                if check_lables != [True, True]:
                    mask = torch.empty(1, 1, H, W)
                    masks[:, :, :, :] = False
                    mask = masks[0].cpu().numpy()
                    final_logit = round(random.uniform(0, 0.1), 2)
                else:
                    mask = mask_per_label[0]
                    final_logit = logit_per_label[0]
            elif self.mode == "andnot":
                if check_lables != [True, False]:
                    mask = masks = torch.empty(1, 1, H, W)
                    masks[:, :, :, :] = False
                    mask = masks[0].cpu().numpy()
                    final_logit = round(random.uniform(0, 0.1), 2)
                else:
                    mask = mask_per_label[0]
                    final_logit = logit_per_label[0]

            masked_image, mask = self.show_mask(mask, image)
            activations.append(final_logit)
            masks_out.append(mask)
            images_out.append(image)
            masked_images.append(Image.fromarray((255 * masked_image).astype(np.uint8)))

        return activations, masks_out, images_out, masked_images
