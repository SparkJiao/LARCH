import gc
import random

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import constants
from constants import DUMP_DIR, DEVICE, TEST_BATCH_SIZE, TOT_IMG_NUM, TEST_SUB_BATCH_SIZE, TEST_DATA_LOAD_WORKERS
from datasets.eval_dataset_text_case import EvalDatasetTextCase as EvalDataset
from eval import eval
from knowledge_embed import KnowledgeData
from models.text_encoder import QueryEncoder
import models
from raw_data_fix import RawData, dialog_to_list
from utils import collate_fn_eval_text_case

KnowledgeEncoder = {
    'simple': models.KnowledgeEncoder,
    'reverse': models.KnowledgeEncoderReverse,
    'fuse': models.KnowledgeEncoderReverseFuse,
    'sep': models.KnowledgeEncoderReverseSeparate,
    'act': models.KnowledgeEncoderReverseActivate,
    'bi': models.KnowledgeEncoderBidirectional,
    'bi_g': models.KnowledgeEncoderBidirectionalGate,
    'bi_g_wo_img': models.KnowledgeEncoderBidirectionalGateWoImg,
    'bi_g_wo_que': models.KnowledgeEncoderBidirectionalGateWoQuery
}[constants.KNOWLEDGE_TYPE]

print(f'Query encoder type: {QueryEncoder}')
print(f'Knowledge encoder type: {KnowledgeEncoder}')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


_top_k = 20


class Evaluator:
    def __init__(self):
        random.seed(constants.SEED)
        np.random.seed(constants.SEED)
        torch.manual_seed(constants.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.raw_data = RawData()
        self.knowledge_data = KnowledgeData(self.raw_data)
        self.query_encoder = QueryEncoder(self.raw_data).to(DEVICE)
        self.knowledge_encoder = KnowledgeEncoder(self.knowledge_data).to(DEVICE)

        self.test_dataset = EvalDataset(self.raw_data, self.knowledge_data, 'small_test')
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=TEST_BATCH_SIZE,
                                           shuffle=False, collate_fn=collate_fn_eval_text_case,
                                           num_workers=TEST_DATA_LOAD_WORKERS)

    @staticmethod
    def fold_tensor(x):
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def eval(self):

        # model_file = DUMP_DIR / 'check_points.tar_kd1.0'
        # model_file = DUMP_DIR / 'check_points.tar_kd1.0_dis_attribute'
        # model_file = DUMP_DIR / 'check_points.tar_mhred_att_only'
        # model_file = DUMP_DIR / 'check_points.tar_mhred_emb_att_only'
        model_file = DUMP_DIR / 'check_points.tar_mhred_full_1.0'
        print(f'Load model from {model_file}')
        state = torch.load(model_file)
        self.query_encoder.load_state_dict(state['query_encoder'])
        self.knowledge_encoder.load_state_dict(state['knowledge_encoder'])
        self.query_encoder.eval()
        self.knowledge_encoder.eval()
        self.query_encoder.apply(set_bn_eval)
        self.knowledge_encoder.apply(set_bn_eval)
        # log_writer = open(DUMP_DIR / 'check_points_kd_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_kd_att_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_mhred_att_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_mhred_emb_att_only_full_neg_img1000.log', 'w')
        log_writer = open(DUMP_DIR / 'case_study_mhred1.0_full_neg_img1000.log', 'w')
        print(f'Load model from {model_file}', file=log_writer, flush=True)

        predictions = []

        with torch.no_grad():
            p_5, p_10, p_20, recall_5, recall_10, recall_20, ndcg_5, ndcg_10, ndcg_20 = 0, 0, 0, 0, 0, 0, 0, 0, 0

            print('start')

            for batch_id, data in enumerate(tqdm(self.test_data_loader)):

                dialog_words, dialog_images, dialog_masks, num_pos_products, products, image_files, dialogs = data
                batch_size = dialog_words.size(0)

                dialog_words = dialog_words.to(DEVICE)
                dialog_images = dialog_images.to(DEVICE)
                dialog_masks = dialog_masks.to(DEVICE)

                session_embeddings = self.query_encoder(dialog_words, dialog_images, dialog_masks)

                del dialog_words, dialog_images, dialog_masks

                all_scores = []
                _images, _style_tips, _celebrities, _attributes = products

                for index in range(0, TOT_IMG_NUM, TEST_SUB_BATCH_SIZE):
                    # print('start sub batch data copy')
                    images = _images[:, index:(index + TEST_SUB_BATCH_SIZE)].to(DEVICE)
                    style_tips = _style_tips[:, index:(index + TEST_SUB_BATCH_SIZE)].to(DEVICE)
                    celebrities = _celebrities[:, index:(index + TEST_SUB_BATCH_SIZE)].to(DEVICE)
                    attributes = _attributes[:, index:(index + TEST_SUB_BATCH_SIZE)].to(DEVICE)

                    _image_num = images.size(1)
                    images = self.fold_tensor(images)
                    style_tips = self.fold_tensor(style_tips)
                    celebrities = self.fold_tensor(celebrities)
                    attributes = self.fold_tensor(attributes)
                    _session_embeddings = self.fold_tensor(session_embeddings.unsqueeze(1).expand(-1, _image_num, -1))

                    desired_images = self.knowledge_encoder(_session_embeddings,
                                                            images, style_tips, celebrities, attributes)

                    scores = torch.cosine_similarity(_session_embeddings, desired_images)
                    all_scores.append(scores.detach().reshape(batch_size, _image_num))

                    del images
                    del style_tips
                    del celebrities
                    del attributes
                    del desired_images
                    del scores

                all_scores = torch.cat(all_scores, dim=1).cpu()

                del _images
                del _style_tips
                del _celebrities
                del _attributes
                del products

                # Remove pad
                num_pos_products, num_padding = zip(*num_pos_products)
                assert len(num_pos_products) == len(num_padding) == batch_size
                for _b, _pad_len in enumerate(num_padding):
                    all_scores[_b, -_pad_len:] = torch.zeros(_pad_len).fill_(-10000.0)

                _p_5, _p_10, _p_20, _recall_5, _recall_10, _recall_20, _ndcg_5, _ndcg_10, _ndcg_20 = eval(all_scores, [
                    list(range(num)) for num in num_pos_products])
                _, _top_image_ids = all_scores.topk(_top_k, dim=-1, largest=True, sorted=True)

                for _b, _batch_ids in enumerate(_top_image_ids):
                    dialog_pred = []
                    for _prod in _batch_ids:
                        if _prod < num_pos_products[_b]:
                            positive = 1
                        else:
                            positive = 0
                        dialog_pred.append({
                            'positive': positive,
                            'image_file_name': str(image_files[_b][_prod.item()])
                        })
                    predictions.append({
                        "dialog_predictions": dialog_pred,
                        "dialog": dialog_to_list(dialogs[_b]),
                        "positive_num": num_pos_products[_b]
                    })

                del all_scores

                N = batch_id + 1
                p_5 += _p_5
                p_10 += _p_10
                p_20 += _p_20
                recall_5 += _recall_5
                recall_10 += _recall_10
                recall_20 += _recall_20
                ndcg_5 += _ndcg_5
                ndcg_10 += _ndcg_10
                ndcg_20 += _ndcg_20

                if batch_id % 10 == 0:
                    gc.collect()
                    print('--p_5:', str(p_5 / N), '--p_10:', str(p_10 / N), '--p_20:', str(p_20 / N),
                          '--recall_5:', str(recall_5 / N), '--recall_10:', str(recall_10 / N), '--recall_20:',
                          str(recall_20 / N),
                          '--ndcg_5:', str(ndcg_5 / N), '--ndcg_10:', str(ndcg_10 / N), '--ndcg_20:', str(ndcg_20 / N))

                    print(f'N = {N}', file=log_writer, flush=True)
                    print('--p_5:', str(p_5 / N), '--p_10:', str(p_10 / N), '--p_20:', str(p_20 / N),
                          '--recall_5:', str(recall_5 / N), '--recall_10:', str(recall_10 / N), '--recall_20:',
                          str(recall_20 / N),
                          '--ndcg_5:', str(ndcg_5 / N), '--ndcg_10:', str(ndcg_10 / N), '--ndcg_20:', str(ndcg_20 / N),
                          file=log_writer, flush=True)

            print('================== End ===========================')
            print('--p_5:', str(p_5 / N), '--p_10:', str(p_10 / N), '--p_20:', str(p_20 / N),
                  '--recall_5:', str(recall_5 / N), '--recall_10:', str(recall_10 / N), '--recall_20:',
                  str(recall_20 / N),
                  '--ndcg_5:', str(ndcg_5 / N), '--ndcg_10:', str(ndcg_10 / N), '--ndcg_20:', str(ndcg_20 / N))

            print('================== End ===========================', file=log_writer, flush=True)
            print('--p_5:', str(p_5 / N), '--p_10:', str(p_10 / N), '--p_20:', str(p_20 / N),
                  '--recall_5:', str(recall_5 / N), '--recall_10:', str(recall_10 / N), '--recall_20:',
                  str(recall_20 / N),
                  '--ndcg_5:', str(ndcg_5 / N), '--ndcg_10:', str(ndcg_10 / N), '--ndcg_20:', str(ndcg_20 / N),
                  file=log_writer, flush=True)

            import json
            with open(DUMP_DIR / f'{model_file}_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
