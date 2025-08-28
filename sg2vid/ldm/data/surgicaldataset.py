import os
import cv2
import glob
import numpy as np
import albumentations
import PIL
from PIL import Image
from torch.utils.data import Dataset

class SurgicalDataset(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=128,
                 interpolation="lanczos",
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()

        image_path_ = [item for sublist in [glob.glob(os.path.join(self.data_root, l, "*.jpg")) for l in self.image_paths] for item in sublist]

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        
        self.image_rescaler = albumentations.Resize(height=self.size, width=self.size,
                                                    interpolation=self.interpolation)
        self.segmentation_rescaler = albumentations.Resize(height=self.size, width=self.size,
                                                            interpolation=cv2.INTER_NEAREST)
        self.labels = {
            "image_path_": image_path_
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
        img = self.image_rescaler(image=img)["image"]
        example["image"] = (img / 127.5 - 1.0).astype(np.float32)
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