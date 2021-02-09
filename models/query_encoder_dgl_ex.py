from typing import List

import dgl
import torch
from PIL import Image
from dgl.nn.pytorch import GATConv
from torch import nn
from torchnlp.word_to_vector.glove import GloVe
from torchvision.models import resnet18
from torchvision.transforms import transforms

import constants
from constants import DATA_DIR, DEVICE, DUMP_DIR, GAT_HEAD, GAT_LAYER, GAT_INTER_DIM, ACT_FN, RESIDUAL


class QueryEncoderExpand(nn.Module):
    EMPTY_IMAGE = torch.zeros(3, 64, 64)
    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    def __init__(self, raw_data):
        super().__init__()
        self.raw_data = raw_data
        self.vocab = self.raw_data.dialog_vocab
        self.vocab_size = len(self.vocab)
        self.images = raw_data.images
        pretrained_embedding = self.get_pretrained_embedding()
        self.word_embed = nn.Embedding(self.vocab_size, 300).from_pretrained(pretrained_embedding).to(DEVICE)
        self.resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2]).to(DEVICE)
        # self.image_fc = nn.Linear(2048, 300).to(DEVICE)
        self.image_fc = nn.Linear(2048, 1024)
        self.text_fc = nn.Linear(300, 1024)

        self.apply(self._init_weights)

        # The GATConv has its own initialization method.
        self.gat = nn.ModuleList()
        act_fn = {
            "elu": torch.nn.ELU(),
            "gelu": torch.nn.GELU(),
        }[ACT_FN]
        for i in range(GAT_LAYER):
            if i == 0:
                self.gat.append(GATConv(in_feats=1024, out_feats=GAT_INTER_DIM, num_heads=GAT_HEAD,
                                        feat_drop=constants.GAT_FEAT_DROPOUT, attn_drop=constants.GAT_ATT_DROPOUT,
                                        residual=RESIDUAL, activation=act_fn))
            elif i < GAT_LAYER - 1:
                self.gat.append(GATConv(in_feats=GAT_HEAD * GAT_INTER_DIM, out_feats=GAT_INTER_DIM,
                                        feat_drop=constants.GAT_FEAT_DROPOUT, attn_drop=constants.GAT_ATT_DROPOUT,
                                        num_heads=GAT_HEAD, residual=RESIDUAL, activation=act_fn))
            else:
                self.gat.append(GATConv(in_feats=GAT_HEAD * GAT_INTER_DIM, out_feats=512,
                                        feat_drop=constants.GAT_FEAT_DROPOUT, attn_drop=constants.GAT_ATT_DROPOUT,
                                        num_heads=1, residual=RESIDUAL, activation=None))

        print(f"query encoder using DGL {self.__class__.__name__} is loaded.")
        print(f"query encoder parameters: layer: {GAT_LAYER}\tinter_dim: {GAT_INTER_DIM}\thead_num: {GAT_HEAD}\t"
              f"feature_dropout: {constants.GAT_FEAT_DROPOUT}\tatt_dropout: {constants.GAT_ATT_DROPOUT}\t"
              f"act_fn: {ACT_FN}, residual: {RESIDUAL}")

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_pretrained_embedding(self):
        pretrained_embedding = GloVe(is_include=lambda x: x in self.vocab, cache=str(DUMP_DIR / '.word_vectors_cache'))
        embedding_weights = torch.zeros((self.vocab_size, 300))
        for i, token in enumerate(self.vocab):
            embedding_weights[i] = pretrained_embedding[token]
        return embedding_weights

    def forward(self, graph: dgl.DGLGraph, graph_word_num: List[int], graph_image_num: List[int],
                all_words: torch.Tensor, all_images: List[int],
                sentences: torch.Tensor, sentence_mask: torch.Tensor,
                utterances: torch.Tensor, utterance_mask: torch.Tensor, session_ids: torch.Tensor):

        all_words = all_words.to(DEVICE)
        sentences = sentences.to(DEVICE)
        sentence_mask = sentence_mask.to(DEVICE)
        utterances = utterances.to(DEVICE)
        utterance_mask = utterance_mask.to(DEVICE)
        session_ids = session_ids.to(DEVICE)
        graph = graph.to(DEVICE)

        all_word_embedding = self.word_embed(all_words)

        num_images = len(all_images)
        image_embeddings = []
        if num_images:
            for image_id in all_images:
                image_path = DATA_DIR / 'images' / self.images[image_id]
                if image_id != 0 and image_path.is_file():
                    try:
                        image = Image.open(image_path).convert("RGB")
                        image = QueryEncoderExpand.transform(image)
                    except OSError:
                        image = QueryEncoderExpand.EMPTY_IMAGE
                else:
                    image = QueryEncoderExpand.EMPTY_IMAGE
                image_embeddings.append(image)
            image_embeddings = torch.stack(image_embeddings).to(DEVICE)
            image_embeddings = self.resnet(image_embeddings).view(num_images, -1)
            image_embeddings = self.image_fc(image_embeddings)

        batch, sent_num, word_num = sentences.size()
        sentence_word_embed = self.word_embed(sentences.view(-1, word_num)).reshape(batch, sent_num, word_num, -1)
        sentence_word_embed = (sentence_word_embed * sentence_mask[:, :, :, None]).sum(dim=2)
        valid_word_num = sentence_mask.sum(dim=2, keepdim=True)  # (batch, sent_num, 1)
        valid_sent_num = (valid_word_num.squeeze(-1) > 0).sum(dim=1)  # (batch)
        valid_word_num[valid_word_num == 0] = 1.
        sentence_word_embed = sentence_word_embed / valid_word_num

        # (batch, utt_num=2, sent_num)
        utt_num, utt_sent_num = utterances.size(1), utterances.size(2)
        utterances = utterances.view(batch, utt_num * utt_sent_num)

        utterance_index = utterances[:, :, None].expand(-1, -1, sentence_word_embed.size(-1))
        utterance_embed = sentence_word_embed.gather(dim=1, index=utterance_index)  # (batch, utt_num * utt_sent_num, h)

        utterance_embed = utterance_embed.reshape(batch, utt_num, utt_sent_num, -1)
        utterance_embed = (utterance_embed * utterance_mask[:, :, :, None]).sum(dim=2)  # (batch, utt_num=2, h)
        valid_utt_sent_num = utterance_mask.sum(dim=2, keepdim=True)  # (batch, utt_num=2, 1)
        valid_utt_sent_num[valid_utt_sent_num == 0] = 1.
        utterance_embed = utterance_embed / valid_utt_sent_num

        session_embed = utterance_embed.mean(dim=1, keepdim=True)  # (batch, 1, h)

        all_word_embedding = self.text_fc(all_word_embedding)
        sentence_word_embed = self.text_fc(sentence_word_embed)
        utterance_embed = self.text_fc(utterance_embed)
        session_embed = self.text_fc(session_embed)

        word_offset = 0
        image_offset = 0
        node_features = []
        for batch_id in range(batch):
            node_features.extend([
                all_word_embedding[word_offset: word_offset + graph_word_num[batch_id]],
                image_embeddings[image_offset: image_offset + graph_image_num[batch_id]],
                sentence_word_embed[batch_id, :valid_sent_num[batch_id].item()],
                utterance_embed[batch_id],
                session_embed[batch_id]
            ])
            word_offset = word_offset + graph_word_num[batch_id]
            image_offset = image_offset + graph_image_num[batch_id]
            # print(len(node_features))

        node_features = torch.cat(node_features, dim=0)
        num_nodes = node_features.size(0)

        for layer_idx in range(GAT_LAYER):
            node_features = self.gat[layer_idx](graph, node_features)
            node_features = node_features.reshape(num_nodes, -1)

        session_index = session_ids.unsqueeze(-1).expand(-1, node_features.size(-1))
        session_hidden = node_features.gather(dim=0, index=session_index)

        return session_hidden
