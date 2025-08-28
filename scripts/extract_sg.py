import os
import glob
import cv2
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from PIL import Image
from natsort import natsorted
from joblib import Parallel, delayed
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def create_graph_from_mask(mask: np.ndarray, 
                           num_classes: int,
                           background_label: int = None,
                           max_distance: int = 50,
                           gaussian_blur_kernel_size: int = 5,
                           apply_gaussian_blur: bool = False,
                           morph_kernel_size: int = 2,
                           touch_threshold: int = 3,
                           min_area: int = 10,
                           min_aspect_ratio: float = 0.1,
                           midas_monocular_depth: np.ndarray = None,
                           raft_optical_flow: np.ndarray = None,
                           save_dir: str = None
                           ) -> nx.Graph:
    """
    Create a graph from a segmentation mask with preprocessing to handle noise.

    :param mask: Segmentation mask as a numpy array.
    :param num_classes: Total number of classes, including background.
    :param background_label: Label of the background class, if any.
    :param max_distance: Maximum distance between connected components to add an edge. TODO: Remove?
    :param gaussian_blur_kernel_size: Kernel size for Gaussian blur.
    :param apply_gaussian_blur: Whether to apply Gaussian blur as a noise reduction step.
    :param morph_kernel_size: Kernel size for morphological operations.
    :param touch_threshold: Minimum overlap required to consider components as touching.
    :param min_area: Minimum area for a component to be considered significant.
    :param min_aspect_ratio: Minimum aspect ratio to consider a component significant.
    :param midas_monocular_depth: If provided, will include average depth per node.
    :param raft_optical_flow: If provided, will include average optical flow per node.
    :param save_dir: If provided, will save the graph as a torch.geometric .pt file.
    :return: A NetworkX graph representing the segmented objects and their relationships.
    """

    # Convert the mask to a numpy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Pre-process mask to reduce impact of annotation noise
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)  # Convert mask to uint8 if it's not already

    if apply_gaussian_blur:
        mask = cv2.GaussianBlur(mask, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    # Erosion followed by dilation to remove small components
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    G = nx.Graph()
    image_height, image_width = mask.shape

    components = {}

    # First pass: identify components and add nodes
    for class_id in range(0, num_classes):  # Assuming class IDs start from 0
        if class_id == background_label:
            continue

        # Create a binary mask for the current class
        class_mask = (mask == class_id).astype(np.uint8)

        # Find connected components (instances) within the class
        num_labels, labels_im = cv2.connectedComponents(class_mask)

        for i in range(1, num_labels):

            component_mask = (labels_im == i).astype(np.uint8)

            # Calculate component properties
            ys, xs = np.where(component_mask)
            if len(xs) == 0 or len(ys) == 0:  # Skip empty components
                continue
            area = len(xs)
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            aspect_ratio = (y_max - y_min + 1) / (x_max - x_min + 1)

            if area < min_area or aspect_ratio < min_aspect_ratio:
                continue  # Skip components that don't meet the criteria

            relative_width = (x_max - x_min) / image_width
            relative_height = (y_max - y_min) / image_height
            relative_centroid_x = xs.mean() / image_width
            relative_centroid_y = ys.mean() / image_height

            one_hot_class = [0] * num_classes
            one_hot_class[class_id] = 1
            features = one_hot_class + [relative_width, relative_height, relative_centroid_x, relative_centroid_y]

            if midas_monocular_depth is not None and raft_optical_flow is not None:
                depth_values = midas_monocular_depth[component_mask == 1]
                depth_mean = np.mean(depth_values)

                optical_flow_x_values = raft_optical_flow[0, :, :][component_mask == 1]
                optical_flow_y_values = raft_optical_flow[1, :, :][component_mask == 1]
                optical_flow_x_mean = np.mean(optical_flow_x_values)
                optical_flow_y_mean = np.mean(optical_flow_y_values)
                features = one_hot_class + [relative_width, relative_height, relative_centroid_x, relative_centroid_y, depth_mean, optical_flow_x_mean, optical_flow_y_mean]

            # Add node to the graph
            node_id = len(G.nodes)  # Unique ID for the node
            G.add_node(node_id, features=features, centroid=(relative_centroid_x, relative_centroid_y))
            components[node_id] = component_mask

    # Second pass: add edges between touching components
    for idx1, data1 in G.nodes(data=True):

        for idx2, data2 in G.nodes(data=True):
            if idx1 >= idx2:
                continue

            # Use individual component masks for each node
            component1_mask = components[idx1]
            component2_mask = components[idx2]

            # Check if the dilated components are touching
            dilated_component1 = cv2.dilate(component1_mask, np.ones((3, 3), np.uint8), iterations=1)
            touching = cv2.bitwise_and(dilated_component1, component2_mask)

            # Add an edge if components are touching
            if np.sum(touching) >= touch_threshold:
                G.add_edge(idx1, idx2)

        data1['x'] = data1['features']

    if save_dir is not None:
        torch.save(from_networkx(G), save_dir)
    return G


class SurgicalDataset(Dataset):
    def __init__(self,
                 data_root,
                 video_prefix,
                 video_id,
                 midas_transforms,
                 size=128
                 ):

        self.data_root = data_root
        self.video_prefix = video_prefix
        self.video_id = video_id
        self.midas_transforms = midas_transforms
        self.size = size

        frame_list = natsorted(glob.glob(os.path.join(data_root, "video_frames", video_prefix+video_id, "*.jpg")))
        self.mask_list = [i.replace('.jpg', '.png').replace('video_frames', 'segm_ann') for i in frame_list]
        self.graph_list = [i.replace('.jpg', '.pt').replace('video_frames', 'scene_graph_updated') for i in frame_list] 
        
        # last pair is duplicate of last frame, to compensate length and lack of optical flow and depth
        self.frame_pairs = [(frame_list[i], frame_list[i+1]) for i in range(len(frame_list) - 1)]
        self.frame_pairs.append((frame_list[-1], frame_list[-1]))
        self._length = len(self.frame_pairs)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        frame_path, frame_next_path = self.frame_pairs[i]
        example["frame_path"] = frame_path
        example["frame_next_path"] = frame_next_path
        example["mask_path"] = self.mask_list[i]
        example["graph_path"] = self.graph_list[i]

        frame = Image.open(example["frame_path"]).convert('RGB')
        frame = F.to_tensor(frame)
        frame = F.resize(frame, size=[self.size, self.size], antialias=False)
        example["frame"] = frame
        frame_next = Image.open(example["frame_next_path"]).convert('RGB')
        frame_next = F.to_tensor(frame_next)
        frame_next = F.resize(frame_next, size=[self.size, self.size], antialias=False)
        example["frame_next"] = frame_next

        #TODO: Maybe redundant
        midas_img = cv2.imread(example["frame_path"])
        midas_img = cv2.cvtColor(midas_img, cv2.COLOR_BGR2RGB)
        midas_img = midas_transforms(midas_img)
        example["midas_img"] = midas_img.squeeze(0)

        mask = np.array(Image.open(example["mask_path"])).astype(np.uint8)
        example["mask"] = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return example


def build_loaders(batch, data_root, video_prefix, video_id, midas_transforms, size):
    dataset = SurgicalDataset(
        data_root=data_root,
        video_prefix=video_prefix,
        video_id=video_id,
        midas_transforms=midas_transforms,
        size=size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=32,
        shuffle=False,
    )
    return dataloader


if __name__ == "__main__":

    batch_size = 16
    device = "cuda"
    size = 128

    data_root = "/path/to/dataset/Cholec-80"
    video_prefix = "video"
    num_classes = 13
    background_label = 0

    # data_root = "/path/to/dataset/Cataract-1K"
    # video_prefix = "case_"
    # num_classes = 14
    # background_label = 0

    # data_root = "/path/to/dataset/Cataracts-50"
    # video_prefix = "train"
    # num_classes = 18
    # background_label = 255

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=True).to(device)
    raft_model.eval()

    midas_type = "DPT_Large" # "MiDaS_small"
    midas_model = torch.hub.load("intel-isl/MiDaS", midas_type)
    midas_model.to(device)
    midas_model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transforms = midas_transforms.dpt_transform

    video_list = natsorted(glob.glob(os.path.join(data_root, "video_frames", video_prefix + "*")))

    for video in video_list:
        print(f"Processing video: {os.path.basename(video)}")
        target_dir = video.replace("video_frames", "scene_graph_updated")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        video_id = os.path.basename(video).replace(video_prefix, "")

        dataloader = build_loaders(batch_size, data_root, video_prefix, video_id, midas_transforms, size)
        tqdm_object = tqdm(dataloader, total=len(dataloader))
        for batch in tqdm_object:
            batch = {k: (v if k.endswith("path") else v.to(device)) for k, v in batch.items()}

            if len(batch["mask"]) != batch_size: process_batch_size = len(batch["mask"])
            else: process_batch_size = batch_size

            mask_batch = [batch["mask"][i] for i in range(process_batch_size)]
            frame_batch, frame_next_batch = raft_transforms(batch["frame"], batch["frame_next"])
            
            with torch.no_grad():
                optical_flow = raft_model(frame_batch, frame_next_batch)  
            optical_flow = optical_flow[-1]
            max_norm = torch.sum(optical_flow**2, dim=1).sqrt().max()
            epsilon = torch.finfo((optical_flow).dtype).eps
            normalized_optical_flow = optical_flow / (max_norm + epsilon)
            normalized_optical_flow = [normalized_optical_flow[i].detach().cpu().numpy() for i in range(process_batch_size)]

            with torch.no_grad():
                monocular_depth = midas_model(batch["midas_img"])
            monocular_depth = torch.nn.functional.interpolate(monocular_depth.unsqueeze(1), size=size, mode="bicubic", align_corners=False).squeeze(1)
            monocular_depth = [monocular_depth[i].detach().cpu().numpy() for i in range(process_batch_size)]
            monocular_depth = [(monocular_depth[i] - np.min(monocular_depth[i])) / (np.max(monocular_depth[i]) - np.min(monocular_depth[i])) for i in range(process_batch_size)]

            Parallel(n_jobs=process_batch_size)(delayed(create_graph_from_mask)(mask,
                                                                        num_classes=num_classes,
                                                                        background_label=background_label,
                                                                        morph_kernel_size=2,
                                                                        min_area=25,
                                                                        min_aspect_ratio=0.1,
                                                                        save_dir=graph_path,
                                                                        midas_monocular_depth=monocular_depth[i],
                                                                        raft_optical_flow=normalized_optical_flow[i]) 
                                                                        for i, (mask, graph_path) in enumerate(zip(mask_batch, batch["graph_path"])))

