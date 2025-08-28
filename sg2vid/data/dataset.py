import os, math, random
import json
import glob
import cv2
from natsort import natsorted
from PIL import Image
import numpy as np
from decord import VideoReader

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from diffusers.utils import logging
import albumentations as A
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
from ..graph_encoder.graph_segclip_masked import GraphEncoder

logger = logging.get_logger(__name__)

class SurgicalDataset(Dataset):
    def __init__(
            self,
            video_folder,
            split_mode,
            dataset_name,
            class_size=None,
            ignore_index=None,
            sample_size=256, 
            sample_n_frames=16,
            overlap_size=1,
            train_graph_encoder=False,
            return_graph_emb=False,
            **kwargs,
        ):
        self.video_folder = video_folder
        self.split_mode = split_mode
        self.dataset_name = dataset_name
        self.class_size = class_size
        self.ignore_index = ignore_index
        self.sample_n_frames = sample_n_frames
        self.train_graph_encoder = train_graph_encoder
        self.return_graph_emb = return_graph_emb
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        
        # separate transform bcs pixel_transform can transform multiple frames together.
        # graph transform with albumentation for easier mask handling
        self.pixel_transforms = transforms.Compose([
                                transforms.Resize(sample_size, antialias=None),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        self.graph_transforms = A.Compose([
                                A.Resize(sample_size[0], sample_size[1], interpolation=cv2.INTER_LANCZOS4),
                                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)])

        video_txt = os.path.join(video_folder, "split", dataset_name+"_"+split_mode+".txt")
        with open(video_txt, "r") as f:
            self.video_list = f.read().splitlines()
        if self.dataset_name == "cataract1k":
            text = "sequence during cataract surgery."
        elif self.dataset_name == "cholec80":
            text = "sequence during cholecystectomy surgery."
        elif self.dataset_name == "cataracts50":
            text = "sequence during cataracts surgery."
        else:
            raise NotImplementedError

        self.text = text
        self.sample = []
        for video in self.video_list:
            frame_path = os.path.join(video_folder, "video_frames", video)
            segm_path = os.path.join(video_folder, "segm_ann", video)
            graph_path = os.path.join(video_folder, "scene_graph", video)
            
            frame_list = natsorted(glob.glob(frame_path + "/*.jpg"))
            segm_list = natsorted(glob.glob(segm_path + "/*.png"))
            graph_list = natsorted(glob.glob(graph_path + "/*.pt"))
            
            frame_list = [os.path.basename(x) for x in frame_list]
            
            
            for n in range(0, len(frame_list), overlap_size):
                if n + self.sample_n_frames < len(frame_list):
                    self.sample.append({"video": os.path.basename(video), 
                                        "frame": frame_list[n:n + self.sample_n_frames], 
                                        "segm": segm_list[n:n + self.sample_n_frames], 
                                        "graph": graph_list[n:n + self.sample_n_frames]}) 
        
        if self.return_graph_emb:
            self.embedding_type = kwargs['embedding_type']
            if 'model_masked' in kwargs:
                self.model_masked = GraphEncoder(kwargs["graph_input_dim"], kwargs["graph_hidden_dim"], 
                                                kwargs["graph_embedding_dim"], kwargs["trainable"],
                                                graph_conv_type = kwargs["graph_conv_type"], 
                                                graph_norm_type = kwargs["graph_norm_type"], 
                                                graph_encoder_ckpt = kwargs['model_masked'])
                self.model_masked.eval()
            if 'model_segclip' in kwargs:
                self.model_segclip = GraphEncoder(kwargs["graph_input_dim"], kwargs["graph_hidden_dim"], 
                                                kwargs["graph_embedding_dim"], kwargs["trainable"], 
                                                graph_conv_type = kwargs["graph_conv_type"], 
                                                graph_norm_type = kwargs["graph_norm_type"], 
                                                graph_encoder_ckpt = kwargs['model_segclip'])
                self.model_segclip.eval()

            self.validation_graph_embs = []
            if 'validation_graphs' in kwargs:
                for graph in kwargs['validation_graphs']: 
                    # doing this bcs dont wanna list all 16 graphs in config file
                    all_graph = natsorted(glob.glob(os.path.dirname(graph) + "/*"))
                    index = all_graph.index(graph)
                    graphs = all_graph[index:index+self.sample_n_frames]
                    graphs = [torch.load(i) for i in graphs]
                    graphs = Batch.from_data_list(graphs)

                    graph_embeddings = self.get_embedding(graphs, self.embedding_type)
                    graph_embeddings = graph_embeddings.detach().squeeze()
                    self.validation_graph_embs.append(graph_embeddings)  

        self.validation_seq_paths = []
        if 'validation_first_frames' in kwargs:
            for frame in kwargs['validation_first_frames']:
                all_frame = natsorted(glob.glob(os.path.dirname(frame) + "/*"))
                index = all_frame.index(frame)
                frames = all_frame[index:index+self.sample_n_frames]
                self.validation_seq_paths.append(frames)

    #TODO: Should not be inside the dataset. It wont utilise gpu.
    def get_embedding(self, scene_graph, embedding_type):
        if embedding_type == 'masked':
            graph_embeddings = self.model_masked(scene_graph)
        elif embedding_type == 'segclip': 
            graph_embeddings = self.model_segclip(scene_graph)
        elif embedding_type == 'combined':
            graph_embeddings_masked = self.model_masked(scene_graph)
            graph_embeddings_segclip = self.model_segclip(scene_graph)
            graph_embeddings = torch.cat((graph_embeddings_masked, graph_embeddings_segclip), dim=-1)
        return graph_embeddings

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        items = {}
        frames = []
        graphs = []
        segmentations = []
        class_labels = []
        sequence = self.sample[index]
        
        if self.train_graph_encoder or self.return_graph_emb:
            for idx in range(self.sample_n_frames):
                    graph = torch.load(sequence["graph"][idx])
                    if "x" in graph:
                        graphs.append(graph)
                    else:
                        # dummy graph for the case where the graph is empty 
                        G = nx.Graph()
                        features = [0.0] * (self.class_size + 7)
                        if self.ignore_index in ["first", "last"]:
                            features[self.class_size - 1 if self.ignore_index == "last" else 0] = 1
                        G.add_node(len(G.nodes), features=features, centroid=(0.0, 0.0))
                        G.add_node(len(G.nodes), features=features, centroid=(0.0, 0.0))

                        for idx1, data1 in G.nodes(data=True):
                            for idx2, data2 in G.nodes(data=True):
                                if idx1 >= idx2:
                                    continue
                                G.add_edge(idx1, idx2)
                            data1['x'] = data1['features']

                        graph = from_networkx(G)
                        graphs.append(graph)
            
            graphs = Batch.from_data_list(graphs)
            if self.train_graph_encoder:
                items["graph"] = graphs
            
            if self.return_graph_emb:
                graph_embeddings = self.get_embedding(graphs, self.embedding_type)
                graph_embeddings = graph_embeddings.detach().squeeze()
                items["graph_embedding"] = graph_embeddings

        # for now handling dataset of diffusion and graph encoder separately for simplicity
        if self.train_graph_encoder:
            for idx in range(self.sample_n_frames):
                image = os.path.join(self.video_folder, "video_frames", sequence["video"], sequence["frame"][idx])
                image = np.array(Image.open(image).convert("RGB"))
                segmentation = np.array(Image.open(sequence['segm'][idx]))

                #TODO: Reimplement and apply augemntation on sequence instead of individually.
                image, segmentation = self.graph_transforms(image=image, mask=segmentation).values()
                image = (image / 127.5 - 1.0).astype(np.float32)
                image = torch.from_numpy(image).permute(2, 0, 1)
                frames.append(image)

                #TODO: Handle using ignore index values. Not dataset type!
                if self.dataset_name == "cataracts50":
                    segmentation[segmentation == 255] = self.class_size -1
                segmentation = torch.from_numpy(segmentation.astype(np.int64))

                class_label = torch.zeros(self.class_size, dtype=torch.int64)
                class_label[segmentation.unique().long()] = 1
                class_labels.append(class_label)
                
                segmentation = F.one_hot(segmentation, num_classes=self.class_size)
                segmentation = segmentation.permute(2, 0, 1).to(dtype=torch.float32)
                segmentations.append(segmentation)

            items["image"] = torch.stack(frames)
            items["class_label"] = torch.max(torch.stack(class_labels), dim=0).values
            items["segmentation"] = torch.stack(segmentations)

        else:
            for idx in sequence["frame"]:
                frame_path = os.path.join(self.video_folder, "video_frames", sequence["video"], idx)
                frame = np.array(Image.open(frame_path).convert('RGB'))
                frames.append(frame)

            frames = np.array(frames)
            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.0
            pixel_values = self.pixel_transforms(pixel_values)
            items["pixel_values"] = pixel_values
            items["text"] = self.text
            items["first_frame_path"] = os.path.join(self.video_folder, "video_frames", sequence["video"], sequence["frame"][0])
            items["all_frame_path"] = [os.path.join(self.video_folder, "video_frames", sequence["video"], sequence["frame"][i]) for i in range(len(sequence["frame"]))]
        return items