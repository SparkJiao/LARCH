from pathlib import Path

import torch

SEED = 0

DATA_DIR = Path('/home/share/zhanghaoyu/mkdset')
DUMP_DIR = Path('/home/dell/jiaofangkai/dump_MM2020')

TRAIN_BATCH_SIZE, TRAIN_DATA_LOAD_WORKERS = 200, 16
VALID_BATCH_SIZE, VALID_DATA_LOAD_WORKERS = 100, 4
TEST_BATCH_SIZE, TEST_DATA_LOAD_WORKERS = 20, 12

CONTEXT_SIZE = 2
WORD_CUT_OFF = 4
VALUE_CUT_OFF = 10
USER_SPEAKER, SYS_SPEAKER = 0, 1
PRODUCT_ATTRIBUTES = ["available_sizes", "brand", "care", "color", "color_name", "fit", "gender", "length", "material",
                      "name", "neck", "price", "review", "reviewstars", "size_fit", "sleeves", "style", "taxonomy",
                      "type"]

NFEAT = 300
NHID = 100
NFEAT_OUT = 100
DROPOUT = 0
ALPHA = 0.2
NHEADS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEMORY_SIZE = 512
OUTPUT_SIZE = 512
ENTRY_SIZE = 300

NUM_EPOCH = 100
MAX_STEPS = 35000

PRINT_FREQ = 10
VALID_FREQ = 1000
VALID_BATCH = 200

POS_IMG_NUM, NEG_IMG_NUM, TOT_IMG_NUM = 12, 1000, 1012
NUM_CELEBRITIES = 412
TEST_SUB_BATCH_SIZE = 10

SOS_TOKEN = '</s>'
EOS_TOKEN = '</e>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

SOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]
DIALOG_TEXT_MAX_LEN = 30

""" DGLGraph constants """
MAX_SENTENCE_NUM = 4
MAX_SENTENCE_LEN = 20
ACT_FN = 'elu'
RESIDUAL = True

GRAPH_TYPE = 'simple'

GAT_LAYER = 5
GAT_HEAD = 16
GAT_INTER_DIM = 64
GAT_FEAT_DROPOUT = 0.1
GAT_ATT_DROPOUT = 0.1
IF_FULL_CONNECT = True

QUERY_TYPE = 'expand'

""" Knowledge Disable """

DISABLE_STYLETIPS = False
DISABLE_CELEBRITY = False
DISABLE_ATTRIBUTE = False
IMAGE_ONLY = False


""" Optimizer """
OPTIMIZER = 'adam'
WEIGHT_DECAY = 0.0
EMB_ADD_LAYER_NORM = False

# ------------------ bidirectional gate 1.2
KNOWLEDGE_TYPE = 'bi_g'
BID_DROPOUT = 0.1
BID_ATTN_DROPOUT = 0.1
LEARNING_RATE = 1e-5

# # bidirectional gate 1.2 dis sty
# DISABLE_STYLETIPS = True

# # bidirectional gate 1.2 image only
# IMAGE_ONLY = True


# # bidirectional gate 1.2 dis att
# DISABLE_ATTRIBUTE = True

# # bidirectional gate 1.2 dis cel
# DISABLE_CELEBRITY = True


# # -------------------- bidirectional 2.0 (3 layer of MLP)
# LEARNING_RATE = 1e-5
# KNOWLEDGE_TYPE = 'bi'
# BID_DROPOUT = 0.1
# BID_ATTN_DROPOUT = 0.1

# # ------------------bid gate wo image aware knowledge
# KNOWLEDGE_TYPE = 'bi_g_wo_img'

# # ------------------bid gate wo query aware knowledge
# KNOWLEDGE_TYPE = 'bi_g_wo_que'

# # -------------------- bid gate 1.2 gat layer == 4
# GAT_LAYER = 4

# # -------------------- bid gate 1.2 gat layer == 3
# GAT_LAYER = 3

# # -------------------- bid gate 1.2 gat layer == 3
# GAT_LAYER = 6
