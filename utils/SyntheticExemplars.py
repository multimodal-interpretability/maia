import os
import numpy as np
import base64
from PIL import Image
from collections import defaultdict
from typing import Dict, List

class SyntheticExemplars():
    
    def __init__(self, path2exemplars, path2save, mode, n_exemplars=15, im_size=224):
        self.path2exemplars = path2exemplars
        self.n_exemplars = n_exemplars
        self.path2save = path2save
        self.im_size = im_size
        self.mode = mode

        self.exemplars = {}
        self.activations = {}
        self.thresholds = {}

        exemplars, activations = self.net_dissect()
        self.exemplars[mode] = exemplars
        self.activations[mode] = activations

    def net_dissect(self,im_size=224):
        exp_path = f'{self.path2exemplars}/{self.mode}/'
        activations = np.loadtxt(f'{exp_path}/activations.csv', delimiter=',')
        image_array = np.load(f'{exp_path}/images.npy')
        mask_array = np.load(f'{exp_path}/masks.npy')
        all_images = []
        for unit in range(activations.shape[0]):
            curr_image_list = []
            for exemplar_inx in range(min(activations.shape[1],self.n_exemplars)):
                save_path = os.path.join(self.path2save,'synthetic_exemplars',self.mode,str(unit))
                if os.path.exists(os.path.join(save_path,f'{exemplar_inx}.png')):
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
                else:
                    curr_mask = (mask_array[unit,exemplar_inx]/255)+0.25
                    curr_mask[curr_mask>1]=1
                    curr_image = image_array[unit,exemplar_inx]
                    masked_image = curr_image * curr_mask
                    masked_image =  Image.fromarray(masked_image.astype(np.uint8))
                    os.makedirs(save_path,exist_ok=True)
                    masked_image.save(os.path.join(save_path,f'{exemplar_inx}.png'), format='PNG')
                    with open(os.path.join(save_path,f'{exemplar_inx}.png'), "rb") as image_file:
                        masked_image = base64.b64encode(image_file.read()).decode('utf-8')
                    curr_image_list.append(masked_image)
            all_images.append(curr_image_list)

        return all_images,activations[:,:self.n_exemplars]