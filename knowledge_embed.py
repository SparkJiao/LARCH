import json
from collections import Counter, namedtuple
from typing import List

import torch
from torchnlp.word_to_vector import GloVe
from tqdm import tqdm

from constants import DUMP_DIR, DATA_DIR, PRODUCT_ATTRIBUTES, VALUE_CUT_OFF
from utils import load_pickle, dump_pickle

StyleTipsData = namedtuple('StyleTipsData', ['vocab', 'edges', 'scores', 'embeds'])
CelebrityData = namedtuple('CelebrityData', ['celebrity_id', 'product_id', 'scores'])
AttributeData = namedtuple('AttributeData', ['key_vocab', 'value_vocab', 'key_embeds', 'value_embeds'])


class KnowledgeData:
    def __init__(self, raw_data):
        self.pretrained_embedding = GloVe(is_include=lambda x: x in raw_data.dialog_vocab,
                                          cache=str(DUMP_DIR / '.word_vectors_cache'))

        self.raw_data = raw_data
        self.images = raw_data.images
        self.styletips_data = self._get_styletips_data()
        self.celebrity_data = self._get_celebrity_data()
        self.attribute_data = self._get_attribute_data()

        self.styletips_reverse_vocab = {y: x for x, y in self.styletips_data.vocab.items()}

    def _get_styletips_data(self):
        dumped_styletips_data_file = DUMP_DIR / 'styletips_emb.pkl'
        if dumped_styletips_data_file.is_file():
            return load_pickle(dumped_styletips_data_file)
        vocab, edges, scores, embeds = {}, [], [], [torch.zeros(300)]
        styletips_data_file = DATA_DIR / 'knowledge/styletip/styletips_synset.txt'
        with styletips_data_file.open() as file:
            for line in file:
                products: List[str] = [''] * 2
                products[0], products[1], score = map(lambda x: x.strip(), line.split(','))
                products = list(map(lambda x: x.lower(), products))
                score = int(score)
                for product in products:
                    if product not in vocab:
                        idx = len(vocab) + 1
                        vocab[product] = idx
                        embed = torch.zeros(300)
                        n_words = 0
                        for word in product.split():
                            n_words += 1
                            embed += self.pretrained_embedding[word]
                        embeds.append(embed / n_words if n_words else embed)
                edges.append((vocab[products[0]], vocab[products[1]]))
                edges.append((vocab[products[1]], vocab[products[0]]))
                scores.append(score)
        embeds = torch.stack(embeds)
        styletips_data = StyleTipsData(vocab=vocab, edges=edges, scores=scores, embeds=embeds)
        dump_pickle(styletips_data, dumped_styletips_data_file)
        return styletips_data

    def _get_celebrity_data(self):
        dumped_celebrity_data_file = DUMP_DIR / 'celebrity_emb.pkl'
        if dumped_celebrity_data_file.is_file():
            return load_pickle(dumped_celebrity_data_file)
        celebrity_data_file = DATA_DIR / 'knowledge/celebrity/celebrity_distribution.json'
        celebrity_id, product_id = {}, {}
        with celebrity_data_file.open() as file:
            celebrity_json = json.load(file)
        for celebrity, products in celebrity_json.items():
            celebrity = celebrity.lower()
            if celebrity not in celebrity_id:
                celebrity_id[celebrity] = len(celebrity_id) + 1
            for product in products.keys():
                product = product.lower()
                if product not in product_id:
                    product_id[product] = len(product_id) + 1
        scores = [[0] * (len(celebrity_id) + 1) for _ in range(len(product_id) + 1)]
        for celebrity, products in celebrity_json.items():
            celebrity = celebrity.lower()
            cel_id = celebrity_id[celebrity]
            for product, score in products.items():
                product = product.lower()
                prod_id = product_id[product]
                scores[prod_id][cel_id] = score
        celebrity_data = CelebrityData(celebrity_id=celebrity_id, product_id=product_id, scores=scores)
        dump_pickle(celebrity_data, dumped_celebrity_data_file)
        return celebrity_data

    def _get_attribute_data(self):
        dumped_attribute_data_file = DUMP_DIR / 'attribute_emb.pkl'
        if dumped_attribute_data_file.is_file():
            return load_pickle(dumped_attribute_data_file)
        key_vocab, value_vocab = {attr: idx + 1 for idx, attr in enumerate(PRODUCT_ATTRIBUTES)}, {}
        key_embeds, value_embeds = [torch.zeros(300)], [torch.zeros(300)]
        for key in PRODUCT_ATTRIBUTES:
            embed = torch.zeros(300)
            n_words = 0
            for word in key.split('_'):  # different from initial implementation
                n_words += 1
                embed += self.pretrained_embedding[word]
            key_embeds.append(embed / n_words if n_words else embed)
        counters = {attr: Counter() for attr in PRODUCT_ATTRIBUTES}
        product_data_dir = DATA_DIR / 'knowledge/products/'
        files = [file for file in product_data_dir.iterdir() if file.suffix == '.json']
        for json_file in tqdm(files, 'Processing product test_data'):
            try:
                json_object = json.load(json_file.open())
            except json.decoder.JSONDecodeError:
                continue
            for key, value in json_object.items():
                if key in counters:
                    counters[key.lower()].update([value.lower()])
        for key, counter in counters.items():
            key_id = key_vocab[key]
            values = [word for word, freq in counter.most_common() if freq >= VALUE_CUT_OFF]
            for value in values:
                idx = len(value_vocab) + 1
                value_vocab[(key_id, value)] = idx
                embed = torch.zeros(300)
                n_words = 0
                for word in value.split():
                    n_words += 1
                    embed += self.pretrained_embedding[word]
                value_embeds.append(embed / n_words if n_words else embed)
        key_embeds = torch.stack(key_embeds)
        value_embeds = torch.stack(value_embeds)
        attribute_data = AttributeData(key_vocab, value_vocab, key_embeds, value_embeds)
        dump_pickle(attribute_data, dumped_attribute_data_file)
        return attribute_data

    def get_styletips_data(self, json_object):
        name = json_object.get('name', '').lower()
        type = json_object.get('type', '').lower()
        edges = []
        for edge in self.styletips_data.edges:
            pt1, pt2 = edge
            pt1 = self.styletips_reverse_vocab[pt1]
            pt2 = self.styletips_reverse_vocab[pt2]
            if pt1 in name or pt1 in type or pt2 in name or pt2 in type:
                edges.append(edge)
        return edges

    def get_celebrity_data(self, json_object):
        name = json_object.get('name', '').lower()
        type = json_object.get('type', '').lower()
        scores = []
        for product, id in self.celebrity_data.product_id.items():
            if product in name or product in type:
                scores.append(self.celebrity_data.scores[id])
        return scores

    def get_attribute_data(self, json_object):
        keys = []
        values = []
        for key, value in json_object.items():
            # Note: Only a space is also empty.
            if value is not None and value != '' and value != ' ':
                key = key.lower()
                value = value.lower()
                if key not in self.attribute_data.key_vocab:
                    continue
                key_id = self.attribute_data.key_vocab[key]
                if (key_id, value) in self.attribute_data.value_vocab:
                    value_id = self.attribute_data.value_vocab[(key_id, value)]
                    keys.append(key_id)
                    values.append(value_id)
        return list(zip(keys, values))