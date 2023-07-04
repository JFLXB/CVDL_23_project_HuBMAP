from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import json
import pandas as pd 
from abc import ABC, abstractmethod

def generate_mask(img_data):
    mask = np.zeros((512, 512,3), dtype=np.uint8)

    for group in img_data["annotations"]:
        coordinates = group["coordinates"][0]
        points = np.array(coordinates, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        temp = np.zeros((512,512), dtype=np.uint8)
        if group["type"] == "blood_vessel":
            cv2.fillPoly(temp, [points], color=(255))
            mask[:,:,0] += temp
        elif group['type'] == "glomerulus":
            cv2.fillPoly(temp, [points], color=(255))
            mask[:,:,1] += temp
        else:
            cv2.fillPoly(temp, [points], color=(255))
            mask[:,:,2] += temp
    return mask



class AbstractDataset(ABC, Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.transform_mask = transform
        self.tiles_dicts = self._load_polygons()
        self.meta_df = pd.read_csv(f"{image_dir}/tile_meta.csv")

    def _load_polygons(self):
        with open(f'{self.image_dir}/polygons.jsonl', 'r') as polygons:
            json_list = list(polygons)
        tiles_dicts = [json.loads(json_str)for json_str in json_list]
        return tiles_dicts

    def plot_base(self, idx):
        plt.figure(figsize=(10, 10))

        img_data = self.tiles_dicts[idx]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = Image.open(image_path)

        legend_elements = [mpatches.Patch(color='green', label='glomerulus'),
                           mpatches.Patch(color='red', label='blood vessel'),
                           mpatches.Patch(color='yellow', label='unsure')]

        for entry in img_data["annotations"]:
            if entry["type"] == 'glomerulus':
                color = 'green'
            elif entry["type"] == 'blood_vessel':
                color = 'red'
            else:
                color = 'yellow'

            sublist = entry["coordinates"][0]
            x = []
            y = []

            for datapoint in sublist:
                x.append(datapoint[0])
                y.append(datapoint[1])

            plt.scatter(x, y, s=0)
            plt.fill(x, y, color, alpha=0.5)

        plt.imshow(image)
        plt.title("Tile with annotations")
        plt.legend(handles=legend_elements, loc='upper right')

    #plots a example after transformation i.e. the transformed image and all three masksk
    def plot_example(self, idx):
        img, mask = self[idx]
        img = img.permute(1, 2, 0).numpy()
        mask = mask.permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(img)
        axs[0].set_title("Image")
        axs[1].imshow(mask[:,:,0], cmap='gray')
        axs[1].set_title("blood_vessel mask")
        axs[2].imshow(mask[:,:,1], cmap='gray')
        axs[2].set_title("glomerulus mask")
        axs[3].imshow(mask[:,:,2], cmap='gray')
        axs[3].set_title("unsure mask")

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass



# Dataset for all annotated images,
# index operations return mask in the following format:
# [mask_idx, h, w] where mask_dix coresponds to:
# mask_idx: 0 => blood_vessel
# mask_idx:1 => glomerulus
# mask_idx:2 => unsure
class BaseDataset(AbstractDataset):
    def __init__(self, image_dir, transform=None):
        super().__init__(image_dir, transform)

    def __len__(self):
        return len(self.tiles_dicts)

    def __getitem__(self, index):
        img_data = self.tiles_dicts[index]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform_mask(mask) 
            
        return image, mask
    

def _gen_dict_from_json_list(lst):
    out = dict()
    for json in lst:
        out[json["id"]] = json
    return out


    
# Dataset for all EXPERT annotated images (422 tiles),
# index operations return mask in the following format:
# [mask_idx, h, w] where mask_dix coresponds to:
# mask_idx: 0 => blood_vessel
# mask_idx:1 => glomerulus
# mask_idx:2 => unsure
class ExpertDataset(AbstractDataset):
    def __init__(self, image_dir, transform=None):
        super().__init__(image_dir, transform)
        self.d1_ids = self.meta_df.loc[self.meta_df['dataset'] == 1]['id'].tolist()
        self.id_map = _gen_dict_from_json_list(self.tiles_dicts)

    def __len__(self):
        return len(self.d1_ids)

    def __getitem__(self, index):
        img_data = self.id_map[self.d1_ids[index]]
        image_path = f'{self.image_dir}/train/{img_data["id"]}.tif'
        image = np.asarray(Image.open(image_path))
        mask = generate_mask(img_data)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform_mask(mask) 

        return image, mask
    
