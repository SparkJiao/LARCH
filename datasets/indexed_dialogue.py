import warnings
from collections import defaultdict

import stanfordnlp
import torch
import dgl

from constants import DIALOG_TEXT_MAX_LEN, UNK_ID, PAD_ID, MAX_SENTENCE_LEN, CONTEXT_SIZE, IF_FULL_CONNECT

warnings.filterwarnings("ignore", category=UserWarning)


# class ConcatGraph:
#     nlp = stanfordnlp.Pipeline(use_gpu=False)
#
#     def __init__(self, dialog_vocab, utterances):
#         self.words = set()
#         self.images = set()
#         self.sentences = []
#         self.utterances = []
#         self.word_edges = []
#
#         diag_tokens = []
#         graph_mask = []
#         for utterance in utterances:
#             utt_tokens = []
#             for sentence in utterance.sentences:
#                 sent_tokens = []
#                 if sentence.text:
#                     doc = self.nlp(sentence.text)
#                     for sub_sentence in doc.sentences:
#                         tokens = []
#                         governors = []
#                         for word in sub_sentence.words:
#                             governor = dialog_vocab.get(sub_sentence.words[word.governor - 1].text, 0)
#                             word = dialog_vocab.get(word.text, 0)
#                             tokens.append(word)
#                             governors.append(governor)
#                             self.word_edges.append((word, governor))
#                 # FIXME: Set as `1` in baselines
#                 #  Unfair comparison
#                 self.images.update(sentence.images[:4])
#                 self.utterances[-1].append(len(self.sentences))
#                 self.sentences.append((list(_words), list(sentence.images[:4])))
#         self.words = list(self.words)
#         self.images = list(self.images)


class GraphDGL:
    nlp = stanfordnlp.Pipeline(use_gpu=False)

    def __init__(self, dialog_vocab, utterances):
        self.words = set()
        self.images = set()
        self.sentences = []
        self.utterances = []
        self.word_edges = []

        for utterance in utterances:
            self.utterances.append([])
            _utt_image = utterance.sentences[0].images[:1]
            if len(_utt_image) == 0:
                _utt_image.append(0)

            for sentence in utterance.sentences:
                _words = set()
                if sentence.text:
                    doc = self.nlp(sentence.text)
                    for sub_sentence in doc.sentences:
                        for word in sub_sentence.words:
                            # Index starts from 1
                            governor_index = word.governor
                            word = dialog_vocab.get(word.text, UNK_ID)
                            self.words.update([word])
                            _words.update([word])
                            if governor_index != 0:
                                governor = dialog_vocab.get(sub_sentence.words[governor_index - 1].text, UNK_ID)
                                self.word_edges.append((word, governor))
                # if len(_words) > 0:
                self.utterances[-1].append(len(self.sentences))
                self.sentences.append((list(_words), _utt_image))

            self.images.update(_utt_image)

        self.words = list(self.words)
        self.images = list(self.images)

        u, v = [], []
        word2id = {word_id: i for i, word_id in enumerate(self.words)}
        image2id = {image_id: i + len(word2id) for i, image_id in enumerate(self.images)}

        for a, b in self.word_edges:
            u.append(word2id[a])
            v.append(word2id[b])

        sent2id = {}
        sentences = []
        sentence_masks = []
        sent2sub_nodes = defaultdict(list)
        for sent_id, sentence in enumerate(self.sentences):
            sent_word, sent_img = sentence
            off_sent_id = sent_id + len(word2id) + len(image2id)
            sent2id[sent_id] = off_sent_id
            # sentences.append(off_sent_id)
            for word in sent_word:
                u.append(word2id[word])
                v.append(off_sent_id)
                sent2sub_nodes[sent_id].append(word2id[word])
            for img in sent_img:
                u.append(image2id[img])
                v.append(off_sent_id)
                sent2sub_nodes[sent_id].append(image2id[img])

            # Add a pad token to avoid empty node during forward
            # FIXME: Is there any need to remove this node?
            #  If the empty node is removed, the upper utterance edge should removed.
            #  And if the upper utterance node is also empty, it should also be removed.
            if len(sent_word) == 0:
                sent_word.append(PAD_ID)

            # save words for embedding computing.
            sent_word_mask = [1] * len(sent_word)
            if len(sent_word) < MAX_SENTENCE_LEN:
                sent_word = sent_word + [PAD_ID] * (MAX_SENTENCE_LEN - len(sent_word))
                sent_word_mask = sent_word_mask + [0] * (MAX_SENTENCE_LEN - len(sent_word_mask))
            else:
                sent_word = sent_word[:MAX_SENTENCE_LEN]
                sent_word_mask = sent_word_mask[:MAX_SENTENCE_LEN]
            sentences.append(sent_word)
            sentence_masks.append(sent_word_mask)

        self.sentences = sentences
        self.sentence_masks = sentence_masks

        utterances = []
        utterance_masks = []
        utt2id = {}
        utt2sub_nodes = defaultdict(list)
        for utt_id, utterance in enumerate(self.utterances):
            # FIXME: the utterance may be empty (the first utterance). Remove the node?
            assert len(self.utterances) == CONTEXT_SIZE
            off_utt_id = utt_id + len(word2id) + len(image2id) + len(sent2id)
            # utterances.append(off_utt_id)
            utt2id[utt_id] = off_utt_id

            for sent_id in utterance:
                u.append(sent2id[sent_id])
                v.append(off_utt_id)
                utt2sub_nodes[utt_id].append(sent2id[sent_id])

            if IF_FULL_CONNECT:
                _utt_sub_nodes = set()
                for sent_id in utterance:
                    _utt_sub_nodes.update(sent2sub_nodes[sent_id])
                _utt_sub_nodes = list(_utt_sub_nodes)

                u += _utt_sub_nodes
                v += [off_utt_id] * len(_utt_sub_nodes)

                utt2sub_nodes[utt_id].extend(_utt_sub_nodes)

            utt_sent_mask = [1] * len(utterance)

            utterances.append(utterance)
            utterance_masks.append(utt_sent_mask)

        self.utterances = utterances
        self.utterance_masks = utterance_masks

        # padding for utterance in the sentence dim
        max_utt_sent_num = max(map(len, utterances))
        for utt_id, sentences in enumerate(utterances):
            if len(sentences) < max_utt_sent_num:
                sentences += [0] * (max_utt_sent_num - len(sentences))
                utterance_masks[utt_id] += [0] * (max_utt_sent_num - len(utterance_masks[utt_id]))

        session_id = len(utterances) + len(word2id) + len(image2id) + len(sent2id)
        for utt_node_id in utt2id.values():
            u.append(session_id)
            v.append(utt_node_id)

        if IF_FULL_CONNECT:
            _session_sub_nodes = set()
            for utt_sub_nodes in utt2sub_nodes.values():
                _session_sub_nodes.update(utt_sub_nodes)
            _session_sub_nodes = list(_session_sub_nodes)

            u += _session_sub_nodes
            v += [session_id] * len(_session_sub_nodes)

        self.session_id = session_id

        # bidirectional edges
        src = torch.tensor(u + v)
        dst = torch.tensor(v + u)

        # 0-in-degree check
        # # word
        # for word_node_id in word2id.values():
        #     assert word_node_id in src and word_node_id in dst
        # # image
        # for image_node_id in image2id.values():
        #     assert image_node_id in src and image_node_id in dst
        # # sentence
        # for sent_node_id in sent2id.values():
        #     assert sent_node_id in src and sent_node_id in dst
        # # utterance
        # for utt_node_id in utt2id.values():
        #     assert utt_node_id in src and utt_node_id in dst
        # # session
        # assert session_id in src and session_id in dst

        self.graph = dgl.graph((src, dst))
        assert self.graph.nodes().size(0) == session_id + 1, (self.graph.nodes().size(0), session_id + 1)
        # print(self.graph.nodes().size(0))


class SparseGraph:
    nlp = stanfordnlp.Pipeline(use_gpu=False)

    def __init__(self, dialog_vocab, utterances):
        self.words = set()
        self.images = set()
        self.sentences = []
        self.utterances = []
        self.word_edges = []

        utterance_images = []
        for utterance in utterances:
            self.utterances.append([])
            _utt_image = utterance.sentences[0].images[:1]
            if len(_utt_image) == 0:
                _utt_image.append(0)
            utterance_images.append(_utt_image)

            for sentence in utterance.sentences:
                _words = set()
                if sentence.text:
                    doc = self.nlp(sentence.text)
                    for sub_sentence in doc.sentences:
                        for word in sub_sentence.words:
                            # Index starts from 1
                            governor_index = word.governor
                            word = dialog_vocab.get(word.text, UNK_ID)
                            self.words.update([word])
                            _words.update([word])
                            if governor_index != 0:
                                governor = dialog_vocab.get(sub_sentence.words[governor_index - 1].text, UNK_ID)
                                self.word_edges.append((word, governor))
                # if len(_words) > 0:
                self.utterances[-1].append(len(self.sentences))
                self.sentences.append((list(_words), _utt_image))

            self.images.update(_utt_image)

        self.words = list(self.words)
        self.images = list(self.images)

        u, v = [], []
        word2id = {word_id: i for i, word_id in enumerate(self.words)}
        image2id = {image_id: i + len(word2id) for i, image_id in enumerate(self.images)}

        for a, b in self.word_edges:
            u.append(word2id[a])
            v.append(word2id[b])

        sent2id = {}
        sentences = []
        sentence_masks = []
        # sent2sub_nodes = defaultdict(list)
        for sent_id, sentence in enumerate(self.sentences):
            sent_word, sent_img = sentence
            off_sent_id = sent_id + len(word2id) + len(image2id)
            sent2id[sent_id] = off_sent_id
            # sentences.append(off_sent_id)
            for word in sent_word:
                u.append(word2id[word])
                v.append(off_sent_id)

            # for img in sent_img:
            #     u.append(image2id[img])
            #     v.append(off_sent_id)
            #     sent2sub_nodes[sent_id].append(image2id[img])

            # Add a pad token to avoid empty node during forward
            # FIXME: Is there any need to remove this node?
            #  If the empty node is removed, the upper utterance edge should removed.
            #  And if the upper utterance node is also empty, it should also be removed.
            if len(sent_word) == 0:
                sent_word.append(PAD_ID)

            # save words for embedding computing.
            sent_word_mask = [1] * len(sent_word)
            if len(sent_word) < MAX_SENTENCE_LEN:
                sent_word = sent_word + [PAD_ID] * (MAX_SENTENCE_LEN - len(sent_word))
                sent_word_mask = sent_word_mask + [0] * (MAX_SENTENCE_LEN - len(sent_word_mask))
            else:
                sent_word = sent_word[:MAX_SENTENCE_LEN]
                sent_word_mask = sent_word_mask[:MAX_SENTENCE_LEN]
            sentences.append(sent_word)
            sentence_masks.append(sent_word_mask)

        self.sentences = sentences
        self.sentence_masks = sentence_masks

        utterances = []
        utterance_masks = []
        utt2id = {}
        utt2sub_nodes = defaultdict(list)
        for utt_id, utterance in enumerate(self.utterances):
            # FIXME: the utterance may be empty (the first utterance). Remove the node?
            assert len(self.utterances) == CONTEXT_SIZE
            off_utt_id = utt_id + len(word2id) + len(image2id) + len(sent2id)
            # utterances.append(off_utt_id)
            utt2id[utt_id] = off_utt_id

            for sent_id in utterance:
                u.append(sent2id[sent_id])
                v.append(off_utt_id)
                utt2sub_nodes[utt_id].append(sent2id[sent_id])

            for _utterance_image in utterance_images:
                for image_id in _utterance_image:
                    u.append(image2id[image_id])
                    v.append(off_utt_id)
                    utt2sub_nodes[utt_id].append(image2id[image_id])

            utt_sent_mask = [1] * len(utterance)

            utterances.append(utterance)
            utterance_masks.append(utt_sent_mask)

        self.utterances = utterances
        self.utterance_masks = utterance_masks

        # padding for utterance in the sentence dim
        max_utt_sent_num = max(map(len, utterances))
        for utt_id, sentences in enumerate(utterances):
            if len(sentences) < max_utt_sent_num:
                sentences += [0] * (max_utt_sent_num - len(sentences))
                utterance_masks[utt_id] += [0] * (max_utt_sent_num - len(utterance_masks[utt_id]))

        session_id = len(utterances) + len(word2id) + len(image2id) + len(sent2id)
        for utt_node_id in utt2id.values():
            u.append(session_id)
            v.append(utt_node_id)

        _session_sub_nodes = set()
        for utt_sub_nodes in utt2sub_nodes.values():
            _session_sub_nodes.update(utt_sub_nodes)
        _session_sub_nodes = list(_session_sub_nodes)

        u += _session_sub_nodes
        v += [session_id] * len(_session_sub_nodes)

        self.session_id = session_id

        # bidirectional edges
        src = torch.tensor(u + v)
        dst = torch.tensor(v + u)

        self.graph = dgl.graph((src, dst))
        assert self.graph.nodes().size(0) == session_id + 1, (self.graph.nodes().size(0), session_id + 1)


class IndexedDialogue:
    nlp = stanfordnlp.Pipeline(use_gpu=False)

    def __init__(self, dialog_vocab, utterances):
        dialog = []
        images = []
        masks = []

        for utterance in utterances:
            utt_words = []
            utt_images = []
            for sentence in utterance.sentences:
                if sentence.text:
                    doc = self.nlp(sentence.text)
                    for sub_sentence in doc.sentences:
                        for word in sub_sentence.words:
                            # FIXME: No lower() operation?
                            word = dialog_vocab.get(word.text, UNK_ID)
                            utt_words.append(word)
            _utt_image = utterance.sentences[0].images[:1]
            if len(_utt_image) == 0:
                _utt_image.append(0)
            utt_images.append(_utt_image)

            mask = [0] * len(utt_words)
            if len(utt_words) < DIALOG_TEXT_MAX_LEN:
                padding_len = DIALOG_TEXT_MAX_LEN - len(utt_words)
                utt_words += [PAD_ID] * padding_len
                mask += [1] * padding_len
            elif len(utt_words) > DIALOG_TEXT_MAX_LEN:
                utt_words = utt_words[:DIALOG_TEXT_MAX_LEN]
                mask = mask[:DIALOG_TEXT_MAX_LEN]

            dialog.append(utt_words)
            images.append(utt_images)
            masks.append(mask)

        self.words = dialog
        self.images = images
        self.masks = masks
