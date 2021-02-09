import json
import random
from os.path import splitext
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import transforms

from constants import DATA_DIR, GRAPH_TYPE
from knowledge_embed import KnowledgeData
from raw_data_fix import RawData
from .indexed_dialogue import GraphDGL, SparseGraph

GraphClass = {
    "simple": GraphDGL,
    "sparse": SparseGraph
}[GRAPH_TYPE]

print(f"Graph type: {GraphClass}")


class TrainDatasetDGL(BaseDataset):
    EMPTY_IMAGE = torch.zeros(3, 64, 64)
    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    def __init__(self, raw_data: RawData, knowledge_data: KnowledgeData, mode='train'):
        self.raw_data = raw_data
        self.knowledge_data = knowledge_data
        if mode == 'train':
            self.dialogs = raw_data.train_dialogs
        elif mode == 'valid':
            self.dialogs = raw_data.valid_dialogs

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index: int):
        dialog = self.dialogs[index]
        graph = GraphClass(self.raw_data.dialog_vocab, dialog.utterances)

        image = None
        product_file = None
        counter = 0
        while image is None and counter < 10:
            image = random.choice(dialog.pos_images)
            image_name = self.raw_data.images[image]
            product_file = DATA_DIR / 'knowledge/products' / (splitext(image_name)[0] + '.json')

            image_file = DATA_DIR / 'images' / image_name
            if image_file.is_file():
                try:
                    image = Image.open(image_file).convert("RGB")
                    image = TrainDatasetDGL.transform(image)
                except OSError:
                    image = None
            else:
                image = None
            counter += 1
        pos_image = image
        if pos_image is None:
            pos_image = TrainDatasetDGL.EMPTY_IMAGE
            product_file = DATA_DIR / 'knowledge/products' / 'non_file'

        json_object = {}
        if product_file.is_file():
            try:
                json_object = json.load(product_file.open())
            except json.decoder.JSONDecodeError:
                pass
        pos_styletip = self.knowledge_data.get_styletips_data(json_object)
        pos_celebrity = self.knowledge_data.get_celebrity_data(json_object)
        pos_attribute = self.knowledge_data.get_attribute_data(json_object)

        image = None
        product_file = None
        counter = 0
        while image is None and counter < 10:
            image = random.choice(dialog.neg_images)
            image_name = self.raw_data.images[image]
            product_file = DATA_DIR / 'knowledge/products' / (splitext(image_name)[0] + '.json')

            image_file = DATA_DIR / 'images' / image_name
            if image_file.is_file():
                try:
                    image = Image.open(image_file).convert("RGB")
                    image = TrainDatasetDGL.transform(image)
                except OSError:
                    image = None
            else:
                image = None
            counter += 1
        neg_image = image
        if neg_image is None:
            neg_image = TrainDatasetDGL.EMPTY_IMAGE
            product_file = DATA_DIR / 'knowledge/products' / 'non_file'

        json_object = {}
        if product_file.is_file():
            try:
                json_object = json.load(product_file.open())
            except json.decoder.JSONDecodeError:
                pass
        neg_styletip = self.knowledge_data.get_styletips_data(json_object)
        neg_celebrity = self.knowledge_data.get_celebrity_data(json_object)
        neg_attribute = self.knowledge_data.get_attribute_data(json_object)

        return graph.graph, graph.words, graph.images, \
               graph.sentences, graph.sentence_masks, graph.utterances, graph.utterance_masks, graph.session_id, \
               pos_image, pos_styletip, pos_celebrity, pos_attribute, \
               neg_image, neg_styletip, neg_celebrity, neg_attribute
