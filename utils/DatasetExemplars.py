import os
import numpy as np
import base64
from PIL import Image
from collections import defaultdict
from typing import Dict, List


class DatasetExemplars():
    """
    A class for performing network dissection on a given neural network model.

    This class analyzes specific layers and units of a neural network model to 
    understand what each unit in a layer has learned. It uses a set of exemplar 
    images to activate the units and stores the resulting activations, along with
    visualizations of the activated regions in the images.

    Attributes
    ----------
    path2exemplars : str
        Path to the directory containing the exemplar images.
    n_exemplars : int
        Number of exemplar images to use for each unit.
    path2save : str
        Path to the directory where the results will be saved.
    unit_dict : dict
        Dict containing path to each unit: model_name->layer_name->neuron_num
        {
        model_name: {
                layer_name: neuron_list
                }
        }
    im_size : int, optional
        Size to which the images will be resized (default is 224).

    Methods
    -------
    net_dissect(layer: str, im_size: int=224)
        Dissects the specified layer of the neural network, analyzing the response
        to the exemplar images and saving visualizations of activated regions.
    """

    def __init__(self, path2exemplars: str, n_exemplars: int, path2save: str, unit_dict: Dict[str, Dict[str, List[int]]], im_size=224, all_units=False):

        """
        Constructs all the necessary attributes for the DatasetExemplars object.

        Parameters
        ----------
        path2exemplars : str
            Path to the directory containing the exemplar images.
        n_exemplars : int
            Number of exemplar images to use for each unit.
        path2save : str
            Path to the directory where the results will be saved.
        unit_dict : dict
            Dict containing path to each unit: model_name->layer_name->neuron_num
            {
            model_name: {
                    layer_name: neuron_list
                    }
            }
        im_size : int, optional
            Size to which the images will be resized (default is 224).
        """

        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.unit_dict = unit_dict
        self.im_size = im_size
        self.exemplars = defaultdict(lambda : defaultdict(dict))
        self.activations = defaultdict(lambda : defaultdict(dict))
        self.thresholds = defaultdict(lambda : defaultdict(dict))
        for model_name, layers in unit_dict.items():
            for layer in layers:
                if not all_units:
                    exemplars, activations, thresholds = self.net_dissect(model_name, layer, unit_dict[model_name][layer])
                else:
                    exemplars, activations, thresholds = self.net_dissect(model_name, layer, None)
                self.exemplars[model_name][layer] = exemplars
                self.activations[model_name][layer] = activations
                self.thresholds[model_name][layer] = thresholds

    # TODO - Extract image masking
    def net_dissect(self, model_name, layer, units, im_size=224):

        """
        Dissects the specified layer of the neural network.

        This method analyzes the response of units in the specified layer to
        the exemplar images. It generates and saves visualizations of the
        activated regions in these images.

        Parameters
        ----------
        model_name : str
            The name of the model containing the layer to be dissected
        layer : str
            The name of the layer to be dissected.
        im_size : int, optional
            Size to which the images will be resized for visualization 
            (default is 224).

        Returns
        -------
        tuple
            A tuple containing lists of images, activations, and thresholds 
            for the specified layer. The images are Base64 encoded strings.
        """

        exp_path = f'{self.path2exemplars}/{model_name}/imagenet/{layer}'
        activations = np.loadtxt(f'{exp_path}/activations.csv', delimiter=',') # units * exemplars
        thresholds = np.loadtxt(f'{exp_path}/thresholds.csv', delimiter=',') # units
        image_array = np.load(f'{exp_path}/images.npy')
        mask_array = np.load(f'{exp_path}/masks.npy')
        all_images = []
        for unit in range(activations.shape[0]):
            curr_image_list = []
            if units!=None and not(unit in units):
                all_images.append(curr_image_list)
                continue
            for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                save_path = os.path.join(self.path2save,'dataset_exemplars',model_name,layer,str(unit),'netdisect_exemplars')
                if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
                else:
                    curr_mask = np.repeat(mask_array[unit,exemplar_inx], 3, axis=0)
                    curr_image = image_array[unit,exemplar_inx]
                    inside = np.array(curr_mask>0)
                    outside = np.array(curr_mask==0)
                    masked_image = curr_image * inside + 0.1 * curr_image * outside
                    masked_image =  Image.fromarray(np.transpose(masked_image, (1, 2, 0)).astype(np.uint8))
                    masked_image = masked_image.resize([self.im_size, self.im_size], Image.Resampling.LANCZOS)
                    os.makedirs(save_path,exist_ok=True)
                    masked_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
                    # curr_image_list.append({"image": masked_image, "resize": im_size})
            all_images.append(curr_image_list)

        return all_images,activations[:,:self.n_exemplars],thresholds