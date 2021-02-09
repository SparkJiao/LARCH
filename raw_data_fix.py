import json
from collections import Counter, namedtuple
from typing import List

import stanfordnlp
from tqdm import tqdm

from constants import DUMP_DIR, DATA_DIR, USER_SPEAKER, SYS_SPEAKER, CONTEXT_SIZE, WORD_CUT_OFF, \
    SPECIAL_TOKENS
from utils import check_prerequisites, load_pickle, dump_pickle

Dialog = namedtuple('Dialog', ['utterances', 'pos_images', 'neg_images'])
Utterance = namedtuple('Utterance', ['speaker', 'sentences'])
Sentence = namedtuple('Sentence', ['text', 'images'])


def dialog_to_list(dialog: Dialog):
    utterances: List[Utterance] = dialog.utterances
    utt_ls = []
    utt_images = []
    for utt in utterances:
        sentences: List[Sentence] = utt.sentences
        utt_ls.append({
            'text': ' '.join([sent.text for sent in sentences]),
            'images': utt.sentences[0].images[:1]
        })

    return utt_ls


class RawData:
    """
    This script has fixed following bugs in `raw_data.py`:
        - Add special tokens to vocabulary
    """

    def __init__(self):
        # self.url_id, self.images = RawData.get_images()
        self.images = RawData.get_images()
        self.train_dialogs = self._get_dialog_data('train')
        self.valid_dialogs = self._get_dialog_data('valid')
        self.test_dialogs = self._get_dialog_data('test')
        self.small_test_dialogs = self._get_dialog_data('small_test')
        self.dialog_vocab = RawData.get_dialog_vocab(self.train_dialogs + self.valid_dialogs)

    @staticmethod
    def get_images():
        # dumped_url2img_file = DUMP_DIR / 'url2img.pkl'
        dumped_url2img_file = DUMP_DIR / 'url_id2img.pkl'
        if dumped_url2img_file.is_file():
            return load_pickle(dumped_url2img_file)
        # check_prerequisites([dumped_url2img_file], [DUMP_DIR / file for file in
        #                                             ['train_dialogs.pkl', 'valid_dialogs.pkl', 'test_dialogs.pkl']])
        url2img_file = DATA_DIR / 'url2img.txt'
        url_id2img = {}
        with url2img_file.open() as file:
            for line in file:
                line = line.strip()
                if line:
                    url, image = line.split(' ')
                    if url not in url_id2img:
                        url_id2img[url] = image

        image_id_file = DATA_DIR / 'dialogs_final' / 'image_id.json'
        with image_id_file.open() as file:
            url2id = json.load(file)[0]
        for url, id in url2id.items():
            url_id2img[id] = url_id2img.get(url, 'non_file')

        dump_pickle(url_id2img, dumped_url2img_file)
        return url_id2img

        # url_id, images, index = {}, [''], 0
        # with url2img_file.open() as file:
        #     for line in file:
        #         line = line.strip()
        #         if line:
        #             url, image = line.split(' ')
        #             if url not in url_id:
        #                 index += 1
        #                 url_id[url] = index
        #                 images.append(image)
        # dump_pickle((url_id, images), dumped_url2img_file)
        # return url_id, images

    def _get_dialog_data(self, mode):
        dialogs = []
        # dialog_dir = DATA_DIR / 'dialogs_now' / mode
        dialog_dir = DATA_DIR / 'dialogs_final' / mode
        dumped_dialog_file = DUMP_DIR / '{}_dialogs.pkl'.format(mode)
        if dumped_dialog_file.is_file():
            return load_pickle(dumped_dialog_file)

        print('start pre-processing dialogs')

        image_dir = DATA_DIR / 'images'
        files = [file for file in dialog_dir.iterdir() if file.suffix == '.json']
        for json_file in tqdm(files, 'Processing {} data'.format(mode)):
            try:
                json_object = json.load(json_file.open())
            except json.decoder.JSONDecodeError:
                continue
            sentence_stream = []
            for utter_object in json_object:
                speaker = USER_SPEAKER if utter_object.get('speaker') == 'user' else SYS_SPEAKER
                sentence = utter_object.get('utterance', {})
                text = sentence.get('nlg', '')
                text = text.strip().lower() if text is not None else ''
                pos_images = sentence.get('images', [])
                pos_images = pos_images if pos_images is not None else []
                pos_images = [x for x in pos_images if
                              self.images.get(x) and (image_dir / self.images.get(x)).is_file()]
                # pos_images = [self.url_id.get(image, 0) for image in pos_images] if pos_images is not None else []
                # pos_images = [x for x in pos_images if x > 0]
                neg_images = sentence.get('false images', [])
                neg_images = neg_images if neg_images is not None else []
                neg_images = [x for x in neg_images if
                              self.images.get(x) and (image_dir / self.images.get(x)).is_file()]
                # neg_images = [self.url_id.get(image, 0) for image in neg_images] if neg_images is not None else []
                # neg_images = [x for x in neg_images if x > 0]
                sentence_stream.append((speaker, text, pos_images, neg_images))
            utterance_stream = [Utterance(speaker=USER_SPEAKER, sentences=[])]
            for i, (speaker, text, pos_images, neg_images) in enumerate(sentence_stream):
                sentence = Sentence(text=text, images=pos_images)
                if not utterance_stream or speaker != utterance_stream[-1].speaker:
                    utterance_stream.append(Utterance(speaker=speaker, sentences=[sentence]))
                else:
                    utterance_stream[-1].sentences.append(sentence)
                if speaker == SYS_SPEAKER and pos_images and neg_images:
                    utterances = utterance_stream[-(CONTEXT_SIZE + 1):-1]
                    dialogs.append(Dialog(utterances=utterances, pos_images=pos_images, neg_images=neg_images))
        dump_pickle(dialogs, dumped_dialog_file)
        return dialogs

    @staticmethod
    def get_dialog_vocab(dialogs=None):
        dumped_dialog_vocab_file = DUMP_DIR / 'dialog_vocab_special.pkl'
        if dumped_dialog_vocab_file.is_file():
            print(f"Loading processed dialog vocab file from {dumped_dialog_vocab_file}")
            return load_pickle(dumped_dialog_vocab_file)

        # Add special tokens to vocabulary
        words = SPECIAL_TOKENS[:]

        counter = Counter()
        nlp = stanfordnlp.Pipeline(processors='tokenize', lang='en', tokenize_pretokenized=True)
        for dialog in dialogs:
            for utterance in dialog.utterances:
                for sentence in utterance.sentences:
                    if not sentence.text: continue
                    doc = nlp(sentence.text)
                    for _sentence in doc.sentences:
                        # FIXME: No lower() operation?
                        counter.update([token.text for token in _sentence.tokens])
        words += [word for word, freq in counter.most_common() if freq >= WORD_CUT_OFF]
        vocab = {word: wid for wid, word in enumerate(words)}
        assert list(vocab.keys())[:len(SPECIAL_TOKENS)] == SPECIAL_TOKENS
        print(f"Save processed dialogue vocabulary to {dumped_dialog_vocab_file}")
        dump_pickle(vocab, dumped_dialog_vocab_file)
        return vocab
