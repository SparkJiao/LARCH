import itertools
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Iterable
from constants import CONTEXT_SIZE

import torch
import dgl
from dgl import DGLGraph


def check_input_dir(input_dir: Path, contained_dirs: List[str] = None, contained_files: List[str] = None):
    """Check if the input_dir contains all the contained_dirs and contained_files."""
    if not input_dir.is_dir():
        raise Exception("{} is not an existed directory.".format(input_dir.absolute()))
    if not contained_dirs: contained_dirs = []
    if not contained_files: contained_files = []
    for contained_dir in contained_dirs:
        dir = input_dir / contained_dir
        if not (dir).is_dir():
            raise Exception("{} is not an existed directory.".format(dir.absolute()))
    for contained_file in contained_files:
        file = input_dir / contained_file
        if not (file).is_file():
            raise Exception("{} is not an existed file.".format(file.absolute()))


def check_output_dir(output_dir: Path):
    """Check if the output_dir exist."""
    if not output_dir.exists():
        output_dir.mkdir()
    elif not output_dir.is_dir():
        raise Exception("{} is not a directory.".format(output_dir.absolute()))


def dump_pickle(obj, pkl_file: Path):
    """Dump a python object into a .pkl file."""
    logging.info('Dumping to {}... '.format(pkl_file.absolute()))
    with pkl_file.open('wb') as file:
        pickle.dump(obj, file)
    logging.info('{} dumped.'.format(pkl_file.absolute()))


def load_pickle(pkl_file: Path):
    """Read a python object from a .pkl file."""
    logging.info('Loading {}... '.format(pkl_file))
    with pkl_file.open('rb') as file:
        obj = pickle.load(file)
    logging.info('{} loaded.'.format(pkl_file))
    return obj


def check_prerequisites(dependent_files: List[Path], files: List[Path]):
    if not all([file.is_file() for file in dependent_files]) and any([file.is_file() for file in files]):
        raise Exception("{} not exist but {} exist.".format(', '.join(map(str, dependent_files)),
                                                            ', '.join(map(str, files))))


def collate_fn(batch):
    batch_size = len(batch)
    graphs, images, style_tips, celebrity, attributes, labels = zip(*batch)
    graph_words = list(set(itertools.chain.from_iterable([graph.words for graph in graphs])))
    graph_images = list(set(itertools.chain.from_iterable([graph.images for graph in graphs])))
    graph_sentences = [graph.sentences for graph in graphs]
    graph_utterances = [graph.utterances for graph in graphs]
    num_sentences = list(map(len, graph_sentences))
    num_utterances = list(map(len, graph_utterances))
    graph_size = len(graph_words) + len(graph_images) + sum(num_sentences) + sum(num_utterances) + batch_size

    word2nid = {word_id: i for i, word_id in enumerate(graph_words)}
    image2nid = {image_id: i + len(word2nid) for i, image_id in enumerate(graph_images)}
    adjacency_matrix = torch.eye(graph_size)
    current_nid = len(word2nid) + len(image2nid)

    session_nodes = []
    for i, graph in enumerate(graphs):
        session_nid = current_nid + len(graph.sentences) + len(graph.utterances)
        session_nodes.append(session_nid)
        for word, governor in graph.word_edges:
            nid1 = word2nid.get(word)
            nid2 = word2nid.get(governor)
            adjacency_matrix[nid1][nid2] = 1
            adjacency_matrix[nid2][nid1] = 1
        for sentence_ids in graph.utterances:
            utterance_nid = current_nid + len(sentence_ids)
            adjacency_matrix[session_nid][utterance_nid] = 1
            adjacency_matrix[utterance_nid][session_nid] = 1
            for sentence_id in sentence_ids:
                sentence_nid = current_nid
                current_nid += 1
                adjacency_matrix[utterance_nid][sentence_nid] = 1
                adjacency_matrix[sentence_nid][utterance_nid] = 1
                words, _images = graph.sentences[sentence_id]
                for word in words:
                    nid = word2nid.get(word)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
                for image in _images:
                    nid = image2nid.get(image)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
            current_nid += 1
        current_nid += 1
    session_nodes = torch.tensor(session_nodes)

    style_tips_lens = torch.tensor(list(map(len, style_tips)))
    style_tips_len = max(style_tips_lens).item()
    celebrity_lens = torch.tensor(list(map(len, celebrity)))
    celebrity_len = max(celebrity_lens).item()
    attributes_lens = torch.tensor(list(map(len, attributes)))
    attributes_len = max(attributes_lens).item()

    style_tips = [(lst + [(0, 0)] * (style_tips_len - len(lst))) for lst in style_tips]
    if style_tips_len:
        style_tips = torch.stack([torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in style_tips])
    else:
        style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    celebrity = [(lst + [[0.] * 412] * (celebrity_len - len(lst))) for lst in celebrity]
    if celebrity_len:
        celebrity = torch.stack([torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in celebrity])
    else:
        celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    attributes = [(lst + [(0, 0)] * (attributes_len - len(lst))) for lst in attributes]
    if attributes_len:
        attributes = torch.stack([torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in attributes])
    else:
        attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return word2nid, image2nid, adjacency_matrix, session_nodes, images, style_tips, celebrity, attributes, labels


def collate_fn_(batch):
    batch_size = len(batch)
    graphs, pos_images, pos_style_tips, pos_celebrity, pos_attributes, neg_images, neg_style_tips, neg_celebrity, neg_attributes = zip(
        *batch)
    graph_words = list(set(itertools.chain.from_iterable([graph.words for graph in graphs])))
    graph_images = list(set(itertools.chain.from_iterable([graph.images for graph in graphs])))
    graph_sentences = [graph.sentences for graph in graphs]
    graph_utterances = [graph.utterances for graph in graphs]
    num_sentences = list(map(len, graph_sentences))
    num_utterances = list(map(len, graph_utterances))
    graph_size = len(graph_words) + len(graph_images) + sum(num_sentences) + sum(num_utterances) + batch_size

    word2nid = {word_id: i for i, word_id in enumerate(graph_words)}
    image2nid = {image_id: i + len(word2nid) for i, image_id in enumerate(graph_images)}
    adjacency_matrix = torch.eye(graph_size)
    current_nid = len(word2nid) + len(image2nid)

    session_nodes = []
    for i, graph in enumerate(graphs):
        session_nid = current_nid + len(graph.sentences) + len(graph.utterances)
        session_nodes.append(session_nid)
        for word, governor in graph.word_edges:
            nid1 = word2nid.get(word)
            nid2 = word2nid.get(governor)
            adjacency_matrix[nid1][nid2] = 1
            adjacency_matrix[nid2][nid1] = 1
        for sentence_ids in graph.utterances:
            utterance_nid = current_nid + len(sentence_ids)
            adjacency_matrix[session_nid][utterance_nid] = 1
            adjacency_matrix[utterance_nid][session_nid] = 1
            for sentence_id in sentence_ids:
                sentence_nid = current_nid
                current_nid += 1
                adjacency_matrix[utterance_nid][sentence_nid] = 1
                adjacency_matrix[sentence_nid][utterance_nid] = 1
                words, _images = graph.sentences[sentence_id]
                for word in words:
                    nid = word2nid.get(word)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
                for image in _images:
                    nid = image2nid.get(image)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
            current_nid += 1
        current_nid += 1
    session_nodes = torch.tensor(session_nodes)

    pos_style_tips_lens = torch.tensor(list(map(len, pos_style_tips)))
    pos_style_tips_len = max(pos_style_tips_lens).item()
    pos_celebrity_lens = torch.tensor(list(map(len, pos_celebrity)))
    pos_celebrity_len = max(pos_celebrity_lens).item()
    pos_attributes_lens = torch.tensor(list(map(len, pos_attributes)))
    pos_attributes_len = max(pos_attributes_lens).item()

    pos_style_tips = [(lst + [(0, 0)] * (pos_style_tips_len - len(lst))) for lst in pos_style_tips]
    if pos_style_tips_len:
        pos_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_style_tips])
    else:
        pos_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_celebrity = [(lst + [[0.] * 412] * (pos_celebrity_len - len(lst))) for lst in pos_celebrity]
    if pos_celebrity_len:
        pos_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in pos_celebrity])
    else:
        pos_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    pos_attributes = [(lst + [(0, 0)] * (pos_attributes_len - len(lst))) for lst in pos_attributes]
    if pos_attributes_len:
        pos_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_attributes])
    else:
        pos_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_images = torch.stack(pos_images)

    neg_style_tips_lens = torch.tensor(list(map(len, neg_style_tips)))
    neg_style_tips_len = max(neg_style_tips_lens).item()
    neg_celebrity_lens = torch.tensor(list(map(len, neg_celebrity)))
    neg_celebrity_len = max(neg_celebrity_lens).item()
    neg_attributes_lens = torch.tensor(list(map(len, neg_attributes)))
    neg_attributes_len = max(neg_attributes_lens).item()

    neg_style_tips = [(lst + [(0, 0)] * (neg_style_tips_len - len(lst))) for lst in neg_style_tips]
    if neg_style_tips_len:
        neg_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_style_tips])
    else:
        neg_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_celebrity = [(lst + [[0.] * 412] * (neg_celebrity_len - len(lst))) for lst in neg_celebrity]
    if neg_celebrity_len:
        neg_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in neg_celebrity])
    else:
        neg_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    neg_attributes = [(lst + [(0, 0)] * (neg_attributes_len - len(lst))) for lst in neg_attributes]
    if neg_attributes_len:
        neg_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_attributes])
    else:
        neg_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_images = torch.stack(neg_images)

    return word2nid, image2nid, adjacency_matrix, session_nodes, (
        pos_images, pos_style_tips, pos_celebrity, pos_attributes), (
               neg_images, neg_style_tips, neg_celebrity, neg_attributes)


def collate_fn_eval_batch_1(batch):
    batch_size = len(batch)
    graphs, pos_images, pos_style_tips, pos_celebrity, pos_attributes, neg_images, neg_style_tips, neg_celebrity, neg_attributes = zip(
        *batch)
    graph_words = list(set(itertools.chain.from_iterable([graph.words for graph in graphs])))
    graph_images = list(set(itertools.chain.from_iterable([graph.images for graph in graphs])))
    graph_sentences = [graph.sentences for graph in graphs]
    graph_utterances = [graph.utterances for graph in graphs]
    num_sentences = list(map(len, graph_sentences))
    num_utterances = list(map(len, graph_utterances))
    graph_size = len(graph_words) + len(graph_images) + sum(num_sentences) + sum(num_utterances) + batch_size

    word2nid = {word_id: i for i, word_id in enumerate(graph_words)}
    image2nid = {image_id: i + len(word2nid) for i, image_id in enumerate(graph_images)}
    adjacency_matrix = torch.eye(graph_size)
    current_nid = len(word2nid) + len(image2nid)

    session_nodes = []
    for i, graph in enumerate(graphs):
        session_nid = current_nid + len(graph.sentences) + len(graph.utterances)
        session_nodes.append(session_nid)
        for word, governor in graph.word_edges:
            nid1 = word2nid.get(word)
            nid2 = word2nid.get(governor)
            adjacency_matrix[nid1][nid2] = 1
            adjacency_matrix[nid2][nid1] = 1
        for sentence_ids in graph.utterances:
            utterance_nid = current_nid + len(sentence_ids)
            adjacency_matrix[session_nid][utterance_nid] = 1
            adjacency_matrix[utterance_nid][session_nid] = 1
            for sentence_id in sentence_ids:
                sentence_nid = current_nid
                current_nid += 1
                adjacency_matrix[utterance_nid][sentence_nid] = 1
                adjacency_matrix[sentence_nid][utterance_nid] = 1
                words, _images = graph.sentences[sentence_id]
                for word in words:
                    nid = word2nid.get(word)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
                for image in _images:
                    nid = image2nid.get(image)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
            current_nid += 1
        current_nid += 1
    session_nodes = torch.tensor(session_nodes)

    pos_batch_size = len(pos_images[0])
    pos_style_tips_lens = torch.tensor(list(map(len, pos_style_tips[0])))
    pos_style_tips_len = max(pos_style_tips_lens).item()
    pos_celebrity_lens = torch.tensor(list(map(len, pos_celebrity[0])))
    pos_celebrity_len = max(pos_celebrity_lens).item()
    pos_attributes_lens = torch.tensor(list(map(len, pos_attributes[0])))
    pos_attributes_len = max(pos_attributes_lens).item()

    pos_style_tips = [(lst + [(0, 0)] * (pos_style_tips_len - len(lst))) for lst in pos_style_tips[0]]
    if pos_style_tips_len:
        pos_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_style_tips])
    else:
        pos_style_tips = torch.zeros((pos_batch_size, 1, 2), dtype=torch.long)

    pos_celebrity = [(lst + [[0.] * 412] * (pos_celebrity_len - len(lst))) for lst in pos_celebrity[0]]
    if pos_celebrity_len:
        pos_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in pos_celebrity])
    else:
        pos_celebrity = torch.zeros((pos_batch_size, 1, 412), dtype=torch.float)

    pos_attributes = [(lst + [(0, 0)] * (pos_attributes_len - len(lst))) for lst in pos_attributes[0]]
    if pos_attributes_len:
        pos_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_attributes])
    else:
        pos_attributes = torch.zeros((pos_batch_size, 1, 2), dtype=torch.long)

    pos_images = torch.stack(pos_images[0])

    neg_batch_size = len(neg_images[0])
    neg_style_tips_lens = torch.tensor(list(map(len, neg_style_tips[0])))
    neg_style_tips_len = max(neg_style_tips_lens).item()
    neg_celebrity_lens = torch.tensor(list(map(len, neg_celebrity[0])))
    neg_celebrity_len = max(neg_celebrity_lens).item()
    neg_attributes_lens = torch.tensor(list(map(len, neg_attributes[0])))
    neg_attributes_len = max(neg_attributes_lens).item()

    neg_style_tips = [(lst + [(0, 0)] * (neg_style_tips_len - len(lst))) for lst in neg_style_tips[0]]
    if neg_style_tips_len:
        neg_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_style_tips])
    else:
        neg_style_tips = torch.zeros((neg_batch_size, 1, 2), dtype=torch.long)

    neg_celebrity = [(lst + [[0.] * 412] * (neg_celebrity_len - len(lst))) for lst in neg_celebrity[0]]
    if neg_celebrity_len:
        neg_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in neg_celebrity])
    else:
        neg_celebrity = torch.zeros((neg_batch_size, 1, 412), dtype=torch.float)

    neg_attributes = [(lst + [(0, 0)] * (neg_attributes_len - len(lst))) for lst in neg_attributes[0]]
    if neg_attributes_len:
        neg_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_attributes])
    else:
        neg_attributes = torch.zeros((neg_batch_size, 1, 2), dtype=torch.long)

    neg_images = torch.stack(neg_images[0])

    return word2nid, image2nid, adjacency_matrix, session_nodes, (
        pos_images, pos_style_tips, pos_celebrity, pos_attributes), (
               neg_images, neg_style_tips, neg_celebrity, neg_attributes)


def collate_fn_eval_(batch):
    batch_size = len(batch)
    graphs, pos_images, pos_style_tips, pos_celebrity, pos_attributes, neg_images, neg_style_tips, neg_celebrity, neg_attributes = zip(
        *batch)
    graph_words = list(set(itertools.chain.from_iterable([graph.words for graph in graphs])))
    graph_images = list(set(itertools.chain.from_iterable([graph.images for graph in graphs])))
    graph_sentences = [graph.sentences for graph in graphs]
    graph_utterances = [graph.utterances for graph in graphs]
    num_sentences = list(map(len, graph_sentences))
    num_utterances = list(map(len, graph_utterances))
    graph_size = len(graph_words) + len(graph_images) + sum(num_sentences) + sum(num_utterances) + batch_size

    word2nid = {word_id: i for i, word_id in enumerate(graph_words)}
    image2nid = {image_id: i + len(word2nid) for i, image_id in enumerate(graph_images)}
    adjacency_matrix = torch.eye(graph_size)
    current_nid = len(word2nid) + len(image2nid)

    session_nodes = []
    for i, graph in enumerate(graphs):
        session_nid = current_nid + len(graph.sentences) + len(graph.utterances)
        session_nodes.append(session_nid)
        for word, governor in graph.word_edges:
            nid1 = word2nid.get(word)
            nid2 = word2nid.get(governor)
            adjacency_matrix[nid1][nid2] = 1
            adjacency_matrix[nid2][nid1] = 1
        for sentence_ids in graph.utterances:
            utterance_nid = current_nid + len(sentence_ids)
            adjacency_matrix[session_nid][utterance_nid] = 1
            adjacency_matrix[utterance_nid][session_nid] = 1
            for sentence_id in sentence_ids:
                sentence_nid = current_nid
                current_nid += 1
                adjacency_matrix[utterance_nid][sentence_nid] = 1
                adjacency_matrix[sentence_nid][utterance_nid] = 1
                words, _images = graph.sentences[sentence_id]
                for word in words:
                    nid = word2nid.get(word)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
                for image in _images:
                    nid = image2nid.get(image)
                    adjacency_matrix[sentence_nid][nid] = 1
                    adjacency_matrix[nid][sentence_nid] = 1
            current_nid += 1
        current_nid += 1
    session_nodes = torch.tensor(session_nodes)

    return word2nid, image2nid, adjacency_matrix, session_nodes, (
        pos_images, max(map(len, pos_images)), pos_style_tips, pos_celebrity, pos_attributes), (
               neg_images, max(map(len, neg_images)), neg_style_tips, neg_celebrity, neg_attributes)


def collate_fn_text(batch):
    """
    Borrow the source code from `collate_fn_`.
    """

    batch_size = len(batch)
    dialog_words, dialog_images, dialog_masks, \
    pos_images, pos_style_tips, pos_celebrity, pos_attributes, \
    neg_images, neg_style_tips, neg_celebrity, neg_attributes = zip(*batch)

    dialog_words = torch.LongTensor(dialog_words)
    dialog_images = torch.LongTensor(dialog_images)
    dialog_masks = torch.LongTensor(dialog_masks)

    pos_style_tips_lens = torch.tensor(list(map(len, pos_style_tips)))
    pos_style_tips_len = max(pos_style_tips_lens).item()
    pos_celebrity_lens = torch.tensor(list(map(len, pos_celebrity)))
    pos_celebrity_len = max(pos_celebrity_lens).item()
    pos_attributes_lens = torch.tensor(list(map(len, pos_attributes)))
    pos_attributes_len = max(pos_attributes_lens).item()

    pos_style_tips = [(lst + [(0, 0)] * (pos_style_tips_len - len(lst))) for lst in pos_style_tips]
    if pos_style_tips_len:
        pos_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_style_tips])
    else:
        pos_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_celebrity = [(lst + [[0.] * 412] * (pos_celebrity_len - len(lst))) for lst in pos_celebrity]
    if pos_celebrity_len:
        pos_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in pos_celebrity])
    else:
        pos_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    pos_attributes = [(lst + [(0, 0)] * (pos_attributes_len - len(lst))) for lst in pos_attributes]
    if pos_attributes_len:
        pos_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_attributes])
    else:
        pos_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_images = torch.stack(pos_images)

    neg_style_tips_lens = torch.tensor(list(map(len, neg_style_tips)))
    neg_style_tips_len = max(neg_style_tips_lens).item()
    neg_celebrity_lens = torch.tensor(list(map(len, neg_celebrity)))
    neg_celebrity_len = max(neg_celebrity_lens).item()
    neg_attributes_lens = torch.tensor(list(map(len, neg_attributes)))
    neg_attributes_len = max(neg_attributes_lens).item()

    neg_style_tips = [(lst + [(0, 0)] * (neg_style_tips_len - len(lst))) for lst in neg_style_tips]
    if neg_style_tips_len:
        neg_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_style_tips])
    else:
        neg_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_celebrity = [(lst + [[0.] * 412] * (neg_celebrity_len - len(lst))) for lst in neg_celebrity]
    if neg_celebrity_len:
        neg_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in neg_celebrity])
    else:
        neg_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    neg_attributes = [(lst + [(0, 0)] * (neg_attributes_len - len(lst))) for lst in neg_attributes]
    if neg_attributes_len:
        neg_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_attributes])
    else:
        neg_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_images = torch.stack(neg_images)

    return dialog_words, dialog_images, dialog_masks, (
        pos_images, pos_style_tips, pos_celebrity, pos_attributes), (
               neg_images, neg_style_tips, neg_celebrity, neg_attributes)


def collate_fn_dgl(batch):
    batch_size = len(batch)

    graphs, words, images, sentences, sentence_masks, utterances, utterance_masks, session_ids, \
    pos_images, pos_style_tips, pos_celebrity, pos_attributes, \
    neg_images, neg_style_tips, neg_celebrity, neg_attributes = zip(*batch)

    graphs = dgl.batch(graphs)

    graph_word_num = list(map(len, words))
    graph_image_num = list(map(len, images))
    all_words = list(itertools.chain.from_iterable(words))
    all_images = list(itertools.chain.from_iterable(images))
    all_words = torch.LongTensor(all_words)
    # all_images = torch.LongTensor(all_images)  # List is ok.

    max_sent_num = max(map(len, sentences))
    max_sent_word_num = len(sentences[0][0])
    sentence = torch.zeros(batch_size, max_sent_num, max_sent_word_num, dtype=torch.long)
    sentence_mask = torch.zeros(batch_size, max_sent_num, max_sent_word_num)

    for batch_id, sents in enumerate(sentences):
        sentence[batch_id, :len(sents)] = torch.LongTensor(sents)
        sentence_mask[batch_id, :len(sents)] = torch.FloatTensor(sentence_masks[batch_id])

    # batch, utt_num, utt_sent_num
    max_utt_sent_num = 0
    for batch_id, utts in enumerate(utterances):
        max_utt_sent_num = max(max_utt_sent_num, len(utts[0]))  # Has been padded.
    utterance = torch.zeros(batch_size, CONTEXT_SIZE, max_utt_sent_num, dtype=torch.long)
    utterance_mask = torch.zeros(batch_size, CONTEXT_SIZE, max_utt_sent_num)
    for batch_id, utts in enumerate(utterances):
        utterance[batch_id, :CONTEXT_SIZE, :len(utts[0])] = torch.LongTensor(utts)
        utterance_mask[batch_id, :CONTEXT_SIZE, :len(utts[0])] = torch.FloatTensor(utterance_masks[batch_id])

    # sentences = torch.LongTensor(sentences)
    # sentence_masks = torch.FloatTensor(sentence_masks)
    # utterances = torch.LongTensor(utterances)
    # utterance_masks = torch.FloatTensor(utterance_masks)

    # Add offset
    off_session_ids = []
    offset = 0
    for session_id in session_ids:
        off_session_ids.append(session_id + offset)
        offset += (session_id + 1)
    session_ids = torch.LongTensor(off_session_ids)

    pos_style_tips_lens = torch.tensor(list(map(len, pos_style_tips)))
    pos_style_tips_len = max(pos_style_tips_lens).item()
    pos_celebrity_lens = torch.tensor(list(map(len, pos_celebrity)))
    pos_celebrity_len = max(pos_celebrity_lens).item()
    pos_attributes_lens = torch.tensor(list(map(len, pos_attributes)))
    pos_attributes_len = max(pos_attributes_lens).item()

    pos_style_tips = [(lst + [(0, 0)] * (pos_style_tips_len - len(lst))) for lst in pos_style_tips]
    if pos_style_tips_len:
        pos_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_style_tips])
    else:
        pos_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_celebrity = [(lst + [[0.] * 412] * (pos_celebrity_len - len(lst))) for lst in pos_celebrity]
    if pos_celebrity_len:
        pos_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in pos_celebrity])
    else:
        pos_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    pos_attributes = [(lst + [(0, 0)] * (pos_attributes_len - len(lst))) for lst in pos_attributes]
    if pos_attributes_len:
        pos_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in pos_attributes])
    else:
        pos_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    pos_images = torch.stack(pos_images)

    neg_style_tips_lens = torch.tensor(list(map(len, neg_style_tips)))
    neg_style_tips_len = max(neg_style_tips_lens).item()
    neg_celebrity_lens = torch.tensor(list(map(len, neg_celebrity)))
    neg_celebrity_len = max(neg_celebrity_lens).item()
    neg_attributes_lens = torch.tensor(list(map(len, neg_attributes)))
    neg_attributes_len = max(neg_attributes_lens).item()

    neg_style_tips = [(lst + [(0, 0)] * (neg_style_tips_len - len(lst))) for lst in neg_style_tips]
    if neg_style_tips_len:
        neg_style_tips = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_style_tips])
    else:
        neg_style_tips = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_celebrity = [(lst + [[0.] * 412] * (neg_celebrity_len - len(lst))) for lst in neg_celebrity]
    if neg_celebrity_len:
        neg_celebrity = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.float) for x in lst]) for lst in neg_celebrity])
    else:
        neg_celebrity = torch.zeros((batch_size, 1, 412), dtype=torch.float)

    neg_attributes = [(lst + [(0, 0)] * (neg_attributes_len - len(lst))) for lst in neg_attributes]
    if neg_attributes_len:
        neg_attributes = torch.stack(
            [torch.stack([torch.tensor(x, dtype=torch.long) for x in lst]) for lst in neg_attributes])
    else:
        neg_attributes = torch.zeros((batch_size, 1, 2), dtype=torch.long)

    neg_images = torch.stack(neg_images)

    return (graphs, graph_word_num, graph_image_num, all_words, all_images, sentence,
            sentence_mask, utterance, utterance_mask, session_ids), (
               pos_images, pos_style_tips, pos_celebrity, pos_attributes), (
               neg_images, neg_style_tips, neg_celebrity, neg_attributes)


def _build_batch_products(products):
    from constants import TOT_IMG_NUM, NUM_CELEBRITIES

    batch_size = len(products)
    images, style_tips, celebrities, attributes = zip(*products)

    _images = torch.stack(list(map(torch.stack, images)))
    del images

    style_tips_len = max(map(lambda x: max(map(len, x)), style_tips))
    style_tips_len = style_tips_len if style_tips_len > 0 else 1
    celebrities_len = max(map(lambda x: max(map(len, x)), celebrities))
    celebrities_len = celebrities_len if celebrities_len > 0 else 1
    attributes_len = max(map(lambda x: max(map(len, x)), attributes))
    attributes_len = attributes_len if attributes_len > 0 else 1

    _style_tips = torch.zeros((batch_size, TOT_IMG_NUM, style_tips_len, 2), dtype=torch.long)
    for i, images in enumerate(style_tips):
        for j, style_tip in enumerate(images):
            if len(style_tip) == 0:
                continue
            _style_tips[i, j, :len(style_tip)] = torch.LongTensor(style_tip)
            # for k, (item0, item1) in enumerate(style_tip):
            #     _style_tips[i, j, k, 0] = item0
            #     _style_tips[i, j, k, 1] = item1

    _celebrities = torch.zeros((batch_size, TOT_IMG_NUM, celebrities_len, NUM_CELEBRITIES), dtype=torch.float)
    for i, images in enumerate(celebrities):
        for j, celebrity in enumerate(images):
            if len(celebrity) == 0:
                continue
            _celebrities[i, j, :len(celebrity)] = torch.tensor(celebrity)
            # for k, scores in enumerate(celebrity):
            #     _celebrities[i, j, k] = torch.tensor(scores, dtype=torch.float)

    _attributes = torch.zeros((batch_size, TOT_IMG_NUM, attributes_len, 2), dtype=torch.long)
    for i, images in enumerate(attributes):
        for j, attribute in enumerate(images):
            if len(attribute) == 0:
                continue
            _attributes[i, j, :len(attribute)] = torch.LongTensor(attribute)
            # for k, (key, value) in enumerate(attribute):
            #     _attributes[i, j, k, 0] = key
            #     _attributes[i, j, k, 1] = value

    del images, style_tips, celebrities, attributes, products

    return _images, _style_tips, _celebrities, _attributes


def _build_batch_graph_dgl(graph_ls):
    batch_size = len(graph_ls)

    graphs, words, images, sentences, sentence_masks, utterances, utterance_masks, session_ids = zip(*graph_ls)

    graphs = dgl.batch(graphs)

    graph_word_num = list(map(len, words))
    graph_image_num = list(map(len, images))
    all_words = list(itertools.chain.from_iterable(words))
    all_images = list(itertools.chain.from_iterable(images))
    all_words = torch.LongTensor(all_words)
    # all_images = torch.LongTensor(all_images)  # List is ok.

    max_sent_num = max(map(len, sentences))
    max_sent_word_num = len(sentences[0][0])
    sentence = torch.zeros(batch_size, max_sent_num, max_sent_word_num, dtype=torch.long)
    sentence_mask = torch.zeros(batch_size, max_sent_num, max_sent_word_num)

    for batch_id, sents in enumerate(sentences):
        sentence[batch_id, :len(sents)] = torch.LongTensor(sents)
        sentence_mask[batch_id, :len(sents)] = torch.FloatTensor(sentence_masks[batch_id])

    # batch, utt_num, utt_sent_num
    max_utt_sent_num = 0
    for batch_id, utts in enumerate(utterances):
        max_utt_sent_num = max(max_utt_sent_num, len(utts[0]))  # Has been padded.
    utterance = torch.zeros(batch_size, CONTEXT_SIZE, max_utt_sent_num, dtype=torch.long)
    utterance_mask = torch.zeros(batch_size, CONTEXT_SIZE, max_utt_sent_num)
    for batch_id, utts in enumerate(utterances):
        utterance[batch_id, :CONTEXT_SIZE, :len(utts[0])] = torch.LongTensor(utts)
        utterance_mask[batch_id, :CONTEXT_SIZE, :len(utts[0])] = torch.FloatTensor(utterance_masks[batch_id])

    # Add offset
    off_session_ids = []
    offset = 0
    for session_id in session_ids:
        off_session_ids.append(session_id + offset)
        offset += (session_id + 1)
    session_ids = torch.LongTensor(off_session_ids)

    return graphs, graph_word_num, graph_image_num, all_words, all_images, sentence, \
           sentence_mask, utterance, utterance_mask, session_ids


def collate_fn_eval(batch):
    graphs, num_pos_products, products = zip(*batch)

    graphs = _build_batch_graph_dgl(graphs)
    products = _build_batch_products(products)

    return graphs, num_pos_products, products


def collate_fn_eval_case(batch):
    graphs, num_pos_products, products, image_files, dialogs = zip(*batch)

    graphs = _build_batch_graph_dgl(graphs)
    products = _build_batch_products(products)

    return graphs, num_pos_products, products, image_files, dialogs


def collate_fn_eval_text(batch):
    dialog_words, dialog_images, dialog_masks, num_pos_products, products = zip(*batch)

    dialog_words = torch.tensor(dialog_words, dtype=torch.long)
    dialog_images = torch.tensor(dialog_images, dtype=torch.long)
    dialog_masks = torch.tensor(dialog_masks, dtype=torch.long)

    products = _build_batch_products(products)

    return dialog_words, dialog_images, dialog_masks, num_pos_products, products


def collate_fn_eval_text_case(batch):
    dialog_words, dialog_images, dialog_masks, num_pos_products, products, image_files, dialogs = zip(*batch)

    dialog_words = torch.tensor(dialog_words, dtype=torch.long)
    dialog_images = torch.tensor(dialog_images, dtype=torch.long)
    dialog_masks = torch.tensor(dialog_masks, dtype=torch.long)

    products = _build_batch_products(products)

    return dialog_words, dialog_images, dialog_masks, num_pos_products, products, image_files, dialogs
