import os
import cv2
import glob
import numpy as np
import albumentations as A
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset

class SurgicalDataset(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 num_label,
                 size=128,
                 augment=False,
                 interpolation="lanczos",
                 reassign_background=False,
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.num_label = num_label
        self.size = size
        self.augment = augment
        self.reassign_background = reassign_background

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        
        image_path_ = [item for sublist in [glob.glob(os.path.join(self.data_root, "video_frames", l, "*.jpg")) for l in self.image_paths] for item in sublist]
        segmentation_path_ = [os.path.join(data_root, "segm_ann", os.path.basename(os.path.dirname(path)), os.path.basename(path).split(".")[0]+".png" ) for path in image_path_]

        
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        
        if self.augment:
            self.rescaler = A.Compose([A.OneOf([
                                    A.Resize(height=self.size, width=self.size, interpolation=self.interpolation, p=0.7),
                                    A.RandomCrop(height=self.size, width=self.size, p=0.1),
                                    A.Compose([A.Resize(height=int(self.size * 1.50), width=int(self.size * 1.50), interpolation=self.interpolation), A.RandomCrop(height=self.size, width=self.size)], p=0.1),
                                    A.Compose([A.Resize(height=int(self.size * 1.25), width=int(self.size * 1.25), interpolation=self.interpolation), A.RandomCrop(height=self.size, width=self.size)], p=0.1),
                                    ], p=1.0)])

            self.augmentation = A.Compose([A.HorizontalFlip(p=0.20),
                                A.RandomBrightnessContrast(p=0.20),
                                A.RandomGamma(p=0.20)
                                ])
        else:
            self.rescaler = A.Resize(height=self.size, width=self.size, interpolation=self.interpolation)


        self.labels = {
            "image_path_": image_path_,
            "segmentation_path_": segmentation_path_,
        }
        self._length = len(self.labels["image_path_"])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        segmentation = np.array(Image.open(example["segmentation_path_"])).astype(np.uint8)
        img, segmentation = self.rescaler(image=img, mask=segmentation).values()

        if self.augment:
            img, segmentation = self.augmentation(image=img, mask=segmentation).values()
        
        example["image"] = (img / 127.5 - 1.0).astype(np.float32)
        
        #TODO: Not complete implementation for segmentation yet
        if self.reassign_background:
            segmentation[segmentation == 255] = self.num_label -1
        segmentation = np.eye(self.num_label)[segmentation]
        segmentation = torch.tensor(segmentation).to(dtype=torch.float32)
        example["segmentation"] = segmentation
        return example


class Cataract1KTrain(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Cataract1KValidation(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Cataracts50Train(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Cataracts50Validation(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Cholec80Train(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Cholec80Validation(SurgicalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)