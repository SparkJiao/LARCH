import random
from datetime import datetime
from itertools import chain

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader

import constants
import models
from constants import TRAIN_BATCH_SIZE, TRAIN_DATA_LOAD_WORKERS, LEARNING_RATE, NUM_EPOCH, DUMP_DIR, \
    PRINT_FREQ, VALID_FREQ, DEVICE, VALID_BATCH, VALID_BATCH_SIZE, VALID_DATA_LOAD_WORKERS
from datasets.train_dataset_text import TrainDatasetText
from datasets.valid_dataset_text import ValidDatasetText
from knowledge_embed import KnowledgeData
from loss import calc_loss_
from models.text_encoder import QueryEncoder
from raw_data_fix import RawData
from utils import collate_fn_text


KnowledgeEncoder = {
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


class TrainerText:
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

        self.train_dataset = TrainDatasetText(self.raw_data, self.knowledge_data)
        self.valid_dataset = ValidDatasetText(self.raw_data, self.knowledge_data)
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                            shuffle=True, num_workers=TRAIN_DATA_LOAD_WORKERS,
                                            collate_fn=collate_fn_text)
        self.valid_data_loader = DataLoader(self.valid_dataset, batch_size=VALID_BATCH_SIZE,
                                            shuffle=True, num_workers=VALID_DATA_LOAD_WORKERS,
                                            collate_fn=collate_fn_text)

        self.writer = SummaryWriter(DUMP_DIR / 'tensorboard_mhred_full_1.0')

        self.param_dict = {
            "optimizer": constants.OPTIMIZER,
            "lr": LEARNING_RATE,
            "weight decay": constants.WEIGHT_DECAY,
            "query type": constants.QUERY_TYPE,
            "knowledge type": constants.KNOWLEDGE_TYPE,
            "graph type": constants.GRAPH_TYPE,
            "graph layer": constants.GAT_LAYER,
            "disable style_tips": constants.DISABLE_STYLETIPS,
            "disable attribute": constants.DISABLE_ATTRIBUTE,
            "disable celebrity": constants.DISABLE_CELEBRITY,
            "image only": constants.IMAGE_ONLY
        }

    def train(self):
        self.query_encoder.train()
        self.knowledge_encoder.train()

        self.query_encoder.apply(set_bn_eval)
        self.knowledge_encoder.apply(set_bn_eval)

        epoch_id = 0
        min_valid_loss = None

        params = list(chain.from_iterable([list(model.parameters()) for model in [
            self.query_encoder,
            self.knowledge_encoder,
        ]]))
        params = [param for param in params if param.requires_grad]

        # Optimizer
        if constants.OPTIMIZER == 'adam':
            optimizer = Adam(params, lr=LEARNING_RATE, weight_decay=constants.WEIGHT_DECAY)
        elif constants.OPTIMIZER == 'adamw':
            from torch.optim.adamw import AdamW
            optimizer = AdamW(params, lr=LEARNING_RATE, weight_decay=constants.WEIGHT_DECAY)
        else:
            raise RuntimeError()

        model_file = DUMP_DIR / 'check_points.tar_mhred_full_1.0'
        pre_tot_steps = -1
        tot_steps = 0
        if model_file.is_file():
            print(f'Loading resumed model from {model_file}')
            state = torch.load(model_file)
            self.query_encoder.load_state_dict(state['query_encoder'])
            self.knowledge_encoder.load_state_dict(state['knowledge_encoder'])
            optimizer.load_state_dict(state['optimizer'])
            epoch_id = state['epoch_id']
            min_valid_loss = state['min_valid_loss']
            pre_tot_steps = state['total_steps']
            tot_steps = len(self.train_data_loader) * epoch_id

        print('Start training...')
        sum_loss = 0
        for epoch_id in range(epoch_id, NUM_EPOCH):
            for batch_id, data in enumerate(self.train_data_loader):
                if tot_steps <= pre_tot_steps:
                    tot_steps += 1
                    continue

                optimizer.zero_grad()
                dialog_words, dialog_images, dialog_masks, pos_products, neg_products = data
                pos_images, pos_styletips, pos_celebrity, pos_attributes = pos_products
                neg_images, neg_styletips, neg_celebrity, neg_attributes = neg_products

                batch_size = pos_images.size(0)
                sub_batch_size = batch_size // 2
                for i in range(2):
                    _dialog_words = dialog_words[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _dialog_images = dialog_images[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _dialog_masks = dialog_masks[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _session_embeddings = self.query_encoder(_dialog_words, _dialog_images, _dialog_masks)

                    _pos_images = pos_images[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _pos_style_tips = pos_styletips[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _pos_celebrity = pos_celebrity[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _pos_attributes = pos_attributes[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)

                    _pos_desired_images = self.knowledge_encoder(_session_embeddings, _pos_images,
                                                                 _pos_style_tips, _pos_celebrity,
                                                                 _pos_attributes)

                    _neg_images = neg_images[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _neg_style_tips = neg_styletips[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _neg_celebrity = neg_celebrity[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)
                    _neg_attributes = neg_attributes[(i * sub_batch_size): ((i + 1) * sub_batch_size)].to(DEVICE)

                    _neg_desired_images = self.knowledge_encoder(_session_embeddings, _neg_images,
                                                                 _neg_style_tips, _neg_celebrity,
                                                                 _neg_attributes)

                    loss = calc_loss_(_session_embeddings, _pos_desired_images, _neg_desired_images) / 2.0
                    sum_loss += loss.detach()

                    loss.backward()

                # dialog_words = dialog_words.to(DEVICE)
                # dialog_images = dialog_images.to(DEVICE)
                # dialog_masks = dialog_masks.to(DEVICE)
                # session_embeddings = self.query_encoder(dialog_words, dialog_images, dialog_masks)

                # pos_images = pos_images.to(DEVICE)
                # pos_style_tips = pos_styletips.to(DEVICE)
                # pos_celebrity = pos_celebrity.to(DEVICE)
                # pos_attributes = pos_attributes.to(DEVICE)
                #
                # pos_desired_images = self.knowledge_encoder(session_embeddings, pos_images,
                #                                             pos_style_tips, pos_celebrity,
                #                                             pos_attributes)

                # neg_images = neg_images.to(DEVICE)
                # neg_style_tips = neg_styletips.to(DEVICE)
                # neg_celebrity = neg_celebrity.to(DEVICE)
                # neg_attributes = neg_attributes.to(DEVICE)
                #
                # neg_desired_images = self.knowledge_encoder(session_embeddings, neg_images,
                #                                             neg_style_tips, neg_celebrity,
                #                                             neg_attributes)

                # loss = calc_loss_(session_embeddings, pos_desired_images, neg_desired_images)
                # sum_loss += loss.detach()
                #
                # loss.backward()
                optimizer.step()

                tot_steps += 1
                # Print loss every `TrainConfig.print_freq` batches.
                if (batch_id + 1) % PRINT_FREQ == 0:
                    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sum_loss /= PRINT_FREQ
                    print('epoch: {} \tbatch: {} \tloss: {} \ttime: {} \tstep: {}'.format(
                        epoch_id + 1, batch_id + 1, sum_loss, cur_time, tot_steps))
                    self.writer.add_scalar('train/loss', sum_loss, tot_steps)
                    sum_loss = 0

                # Valid every `TrainConfig.valid_freq` batches.
                if tot_steps % VALID_FREQ == 0:
                    valid_loss = 0
                    valid_batch = 0

                    self.query_encoder.eval()
                    self.knowledge_encoder.eval()

                    with torch.no_grad():
                        for batch_id, data in enumerate(self.valid_data_loader):
                            valid_batch += 1
                            if valid_batch >= VALID_BATCH:
                                break
                            dialog_words, dialog_images, dialog_masks, pos_products, neg_products = data
                            pos_images, pos_styletips, pos_celebrity, pos_attributes = pos_products
                            neg_images, neg_styletips, neg_celebrity, neg_attributes = neg_products

                            dialog_words = dialog_words.to(DEVICE)
                            dialog_images = dialog_images.to(DEVICE)
                            dialog_masks = dialog_masks.to(DEVICE)
                            session_embeddings = self.query_encoder(dialog_words, dialog_images, dialog_masks)

                            pos_images = pos_images.to(DEVICE)
                            pos_style_tips = pos_styletips.to(DEVICE)
                            pos_celebrity = pos_celebrity.to(DEVICE)
                            pos_attributes = pos_attributes.to(DEVICE)

                            pos_desired_images = self.knowledge_encoder(session_embeddings, pos_images,
                                                                        pos_style_tips, pos_celebrity,
                                                                        pos_attributes)

                            neg_images = neg_images.to(DEVICE)
                            neg_style_tips = neg_styletips.to(DEVICE)
                            neg_celebrity = neg_celebrity.to(DEVICE)
                            neg_attributes = neg_attributes.to(DEVICE)

                            neg_desired_images = self.knowledge_encoder(session_embeddings, neg_images,
                                                                        neg_style_tips, neg_celebrity,
                                                                        neg_attributes)

                            valid_loss += calc_loss_(session_embeddings, pos_desired_images, neg_desired_images)

                    valid_loss = valid_loss / VALID_BATCH

                    self.query_encoder.train()
                    self.knowledge_encoder.train()
                    self.query_encoder.apply(set_bn_eval)
                    self.knowledge_encoder.apply(set_bn_eval)

                    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print('valid_loss: {} \ttime: {} \tsteps: {}'.format(valid_loss, cur_time, tot_steps))
                    self.writer.add_scalar('valid/loss', valid_loss, tot_steps)

                    # Save current best model.
                    if min_valid_loss is None or valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        save_dict = {
                            'epoch_id': epoch_id,
                            'min_valid_loss': min_valid_loss,
                            'optimizer': optimizer.state_dict(),
                            'query_encoder': self.query_encoder.state_dict(),
                            'knowledge_encoder': self.knowledge_encoder.state_dict(),
                            'total_steps': tot_steps,
                            'param_dict': self.param_dict
                        }

                        # torch.save(save_dict, DUMP_DIR / 'check_points.tar_mhred_att_only')
                        torch.save(save_dict, model_file)
                        print('Best model saved.')

                if tot_steps >= constants.MAX_STEPS:
                    print(f"Exceeding max steps {constants.MAX_STEPS}. Exit.")
                    break

            if tot_steps >= constants.MAX_STEPS:
                break
