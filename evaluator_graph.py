import gc
import random

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import constants
from constants import DUMP_DIR, DEVICE, TEST_BATCH_SIZE, TOT_IMG_NUM, TEST_SUB_BATCH_SIZE, TEST_DATA_LOAD_WORKERS
from datasets.eval_dataset import EvalDatasetGraphDGL
from eval import eval
from knowledge_embed import KnowledgeData
import models
from raw_data_fix import RawData
from utils import collate_fn_eval

QueryEncoder = {
    'simple': models.QueryEncoder,
    'expand': models.QueryEncoderExpand,
    'weighted': models.QueryEncoderExpandWeighted,
    'ex_vgg': models.QueryEncoderExpandVGG
}[constants.QUERY_TYPE]

KnowledgeEncoder = {
    'bi': models.KnowledgeEncoderBidirectional,
    'bi_g': models.KnowledgeEncoderBidirectionalGate,
    'bi_g_wo_img': models.KnowledgeEncoderBidirectionalGateWoImg,
    'bi_g_wo_que': models.KnowledgeEncoderBidirectionalGateWoQuery,
    'bi_g_vgg': models.KnowledgeEncoderBidirectionalGateVGG
}[constants.KNOWLEDGE_TYPE]

print(f'Query encoder type: {QueryEncoder}')
print(f'Knowledge encoder type: {KnowledgeEncoder}')


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


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

        self.test_dataset = EvalDatasetGraphDGL(self.raw_data, self.knowledge_data, 'test')
        # self.test_dataset = EvalDatasetGraphDGL(self.raw_data, self.knowledge_data, 'valid')
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=TEST_BATCH_SIZE,
                                           shuffle=False, collate_fn=collate_fn_eval,
                                           num_workers=TEST_DATA_LOAD_WORKERS)

    def fold_tensor(self, x):
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def eval(self):

        # model_file = DUMP_DIR / 'check_points.tar_kd2.0_att_only'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.5'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.7'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.12'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.12_all'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.12_all_dis_att'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_att_only1.12_all_dis_cel'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.1'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.3'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-dis-cel'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-att-only'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-dis-att'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-image-only'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-dis-sty'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_full_reverse1.2-image-only-rerun'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_reverse_fuse1.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_reverse_sep1.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_reverse_act1.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bidirectional1.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bidirectional1.0_dis_sty'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bidirectional1.0_att_only'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bidirectional1.0_dis_cel'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.0_dis_sty'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.1'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_img_only'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_dis_sty'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_dis_cel'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_dis_att'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bidirectional2.0'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_wo_img'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_wo_que'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_4gat'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_3gat'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_6gat'
        # model_file = DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_vgg'
        model_file = DUMP_DIR / 'check_points.tar_q_weighted_dgl_bid_gate1.0'
        print(f'Load model from {model_file}')
        state = torch.load(model_file)
        self.query_encoder.load_state_dict(state['query_encoder'])
        self.knowledge_encoder.load_state_dict(state['knowledge_encoder'])
        self.query_encoder.eval()
        self.knowledge_encoder.eval()
        self.query_encoder.apply(set_bn_eval)
        self.knowledge_encoder.apply(set_bn_eval)
        # log_writer = open(DUMP_DIR / 'check_points_kd2.0_att_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only1.7_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only1.12_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only1.12_all_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only1.12_all_dis_att_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_att_only1.12_all_dis_cel_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.1_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.3_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_dis_cel_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_att_only_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_dis_att_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_image_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_dis_sty_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_image_only_rerun_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_reverse_fuse1.0_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_reverse_sep1.0_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_reverse_act1.0_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_val_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_full_reverse1.2_image_only_rerun_val_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bidirectional1.0_r_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bidirectional1.0_dis_sty_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bidirectional1.0_att_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bidirectional1.0_dis_cel_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.0_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.0_dis_sty_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.1_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_img_only_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_dis_sty_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_dis_cel_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_dis_att_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_rm_att_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid2.0_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_wo_img_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_wo_que_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_4gat_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_3gat_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points_dgl_bid_gate1.2_6gat_full_neg_img1000.log', 'w')
        # log_writer = open(DUMP_DIR / 'check_points.tar_dgl_bid_gate1.2_vgg_full_neg_img1000.log', 'w')
        log_writer = open(DUMP_DIR / 'check_points.tar_q_weighted_dgl_bid_gate1.0_full_neg_img_1000.log', 'w')
        print(f'Load model from {model_file}', file=log_writer, flush=True)

        with torch.no_grad():
            p_5, p_10, p_20, recall_5, recall_10, recall_20, ndcg_5, ndcg_10, ndcg_20 = 0, 0, 0, 0, 0, 0, 0, 0, 0

            print('start')

            for batch_id, data in enumerate(tqdm(self.test_data_loader)):

                # print('batch data loaded')
                graph_inputs, num_pos_products, products = data

                session_embeddings = self.query_encoder(*graph_inputs)

                all_scores = []
                _images, _style_tips, _celebrities, _attributes = products
                batch_size = _images.size(0)

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
                    if isinstance(desired_images, tuple):
                        images_scores = torch.cosine_similarity(_session_embeddings, desired_images[0])
                        knowledge_scores = torch.cosine_similarity(_session_embeddings, desired_images[1])
                        scores = constants.MIX_SCALAR * images_scores + (1 - constants.MIX_SCALAR) * knowledge_scores
                    else:
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
