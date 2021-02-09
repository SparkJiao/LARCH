import json
import time
from os.path import splitext

import torch
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import transforms

from constants import DATA_DIR, TOT_IMG_NUM, NEG_IMG_NUM, POS_IMG_NUM, GRAPH_TYPE
from knowledge_embed import KnowledgeData
from raw_data_fix import RawData
from .indexed_dialogue import GraphDGL, SparseGraph

GraphClass = {
    "simple": GraphDGL,
    "sparse": SparseGraph
}[GRAPH_TYPE]

print(f"Graph type: {GraphClass}")

EMPTY_IMAGE = torch.zeros(3, 64, 64)
transform = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class EvalDatasetGraphDGLCase(BaseDataset):

    def __init__(self, raw_data: RawData, knowledge_data: KnowledgeData, mode):
        self.raw_data = raw_data
        self.knowledge_data = knowledge_data
        if mode == 'valid':
            self.dialogs = raw_data.valid_dialogs
        elif mode == 'test':
            self.dialogs = raw_data.test_dialogs
            print(len(self.dialogs))
        elif mode == 'small_test':
            self.dialogs = raw_data.small_test_dialogs
            print(len(self.dialogs))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index: int):
        dialog = self.dialogs[index]
        graph = GraphClass(self.raw_data.dialog_vocab, dialog.utterances)

        images = []
        styletips = []
        celebrities = []
        attributes = []
        image_files = []

        # Fix 0927 9:40
        pos_images = dialog.pos_images[:POS_IMG_NUM]
        neg_images = dialog.neg_images[:NEG_IMG_NUM]
        selected_images = (pos_images + neg_images)
        pad_len = (TOT_IMG_NUM - len(selected_images))
        pad_len = max(0, pad_len)
        if pad_len:
            selected_images += [0] * pad_len

        # using time statistics
        image_load_time = 0
        knowledge_load_time = 0

        for image in selected_images:
            _start_time = time.time()

            image_name = self.raw_data.images[image]
            product_file = DATA_DIR / 'knowledge/products' / (splitext(image_name)[0] + '.json')
            image_file = DATA_DIR / 'images' / image_name

            image_files.append(image_file)

            if image_file.is_file():
                try:
                    image = Image.open(image_file).convert("RGB")
                    image = transform(image)
                except OSError:
                    image = None
            else:
                image = None
            if image is None:
                image = EMPTY_IMAGE
                product_file = DATA_DIR / 'knowledge/products' / 'non_file'

            _image_load_end_time = time.time()

            json_object = {}
            if product_file.is_file():
                try:
                    json_object = json.load(product_file.open())
                except json.decoder.JSONDecodeError:
                    pass

            styletip = self.knowledge_data.get_styletips_data(json_object)
            celebrity = self.knowledge_data.get_celebrity_data(json_object)
            attribute = self.knowledge_data.get_attribute_data(json_object)

            images.append(image)
            styletips.append(styletip)
            celebrities.append(celebrity)
            attributes.append(attribute)

            _knowledge_load_end_time = time.time()

            image_load_time += (_image_load_end_time - _start_time)
            knowledge_load_time += (_knowledge_load_end_time - _image_load_end_time)

        return (graph.graph, graph.words, graph.images,
                graph.sentences, graph.sentence_masks, graph.utterances, graph.utterance_masks, graph.session_id), \
               (len(pos_images), pad_len), (images, styletips, celebrities, attributes), image_files, dialog
