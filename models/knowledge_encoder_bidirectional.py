import torch
from torch import nn
from torch.nn import functional
from torchvision.models import resnet18
from torchvision.transforms import transforms

from constants import MEMORY_SIZE, ENTRY_SIZE, DEVICE, DISABLE_STYLETIPS, DISABLE_ATTRIBUTE, DISABLE_CELEBRITY, \
    IMAGE_ONLY, EMB_ADD_LAYER_NORM, BID_DROPOUT, BID_ATTN_DROPOUT
from knowledge_embed import KnowledgeData

entry_dim = ENTRY_SIZE
F = MEMORY_SIZE


class KnowledgeEncoderBidirectional(nn.Module):
    EMPTY_IMAGE = torch.zeros(3, 64, 64)
    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    def __init__(self, knowledge: KnowledgeData):
        super().__init__()
        print('Model with knowledge ls loading...')
        self.styletips_embedding = nn.Embedding(len(knowledge.styletips_data.vocab) + 1, entry_dim, padding_idx=0
                                                ).from_pretrained(knowledge.styletips_data.embeds).to(DEVICE)
        self.celebrity_embedding = nn.Linear(len(knowledge.celebrity_data.celebrity_id) + 1, entry_dim).to(DEVICE)
        self.attributes_key_embedding = nn.Embedding(len(knowledge.attribute_data.key_vocab) + 1, entry_dim,
                                                     padding_idx=0).from_pretrained(
            knowledge.attribute_data.key_embeds).to(DEVICE)
        self.attributes_value_embedding = nn.Embedding(len(knowledge.attribute_data.value_vocab) + 1, entry_dim,
                                                       padding_idx=0).from_pretrained(
            knowledge.attribute_data.value_embeds).to(DEVICE)

        if EMB_ADD_LAYER_NORM:
            self.style_layer_norm = nn.LayerNorm(entry_dim, eps=1e-12)
            self.attribute_key_layer_norm = nn.LayerNorm(entry_dim, eps=1e-12)
            self.attribute_value_layer_norm = nn.LayerNorm(entry_dim, eps=1e-12)
            self.celebrity_layer_norm = nn.LayerNorm(entry_dim, eps=1e-12)

        self.query_key1 = nn.Linear(entry_dim, F).to(DEVICE)
        self.query_key2 = nn.Linear(entry_dim, F).to(DEVICE)
        self.query_key3 = nn.Linear(entry_dim, F).to(DEVICE)
        self.image_key1 = nn.Linear(entry_dim, F).to(DEVICE)
        self.image_key2 = nn.Linear(entry_dim, F).to(DEVICE)
        self.image_key3 = nn.Linear(entry_dim, F).to(DEVICE)
        self.value1 = nn.Linear(entry_dim, F).to(DEVICE)
        self.value2 = nn.Linear(entry_dim, F).to(DEVICE)
        self.value3 = nn.Linear(entry_dim, F).to(DEVICE)

        self.resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2]).to(DEVICE)
        self.image_fc = nn.Linear(2048, F).to(DEVICE)

        self.query_o = nn.Linear(F, F).to(DEVICE)
        self.image_o = nn.Linear(F, F).to(DEVICE)

        if BID_DROPOUT > 0:
            self.dropout = nn.Dropout(p=BID_DROPOUT)
        else:
            self.dropout = lambda x: x

        if BID_ATTN_DROPOUT > 0:
            self.attn_dropout = nn.Dropout(p=BID_ATTN_DROPOUT)
        else:
            self.attn_dropout = lambda x: x

        # self.out_mlp = nn.Sequential(
        #     nn.Linear(F * 3, 512 * 6),
        #     nn.ReLU(),
        #     nn.Linear(512 * 6, 512)
        # ).to(DEVICE)
        # Version 2.0
        self.out_mlp = nn.Sequential(
            nn.Linear(F * 3, 512 * 6),
            nn.ReLU(),
            nn.Linear(512 * 6, 512 * 2),
            nn.ReLU(),
            nn.Linear(512 * 2, 512)
        )

        print(f"Knowledge disable: attribute-{DISABLE_ATTRIBUTE}, "
              f"style-tips-{DISABLE_STYLETIPS}, celebrity-{DISABLE_CELEBRITY},"
              f"if image only-{IMAGE_ONLY}\nadd layer normalization: {EMB_ADD_LAYER_NORM},"
              f"dropout : {BID_DROPOUT}, attention dropout: {BID_ATTN_DROPOUT}")

    def forward(self, queries, image, styletips, celebrity, attributes):
        '''
        styletips_data is a [train_batch_size * edge_num * entry_dim] tensor
        celebrity_data is a [train_batch_size * column_vector_num * entry_dim] tensor
        attribute_field_data is a [train_batch_size * attribute_num * entry_dim] tensor
        attribute_value_data is a [train_batch_size * attribute_num * entry_dim] tensor
        image_data is a [train_batch_size * 3 * 64 * 64] tensor
        '''
        batch_size = image.size(0)

        image_embeddings = self.resnet(image).view(batch_size, -1)
        image_embeddings = self.image_fc(image_embeddings).view(batch_size, -1)

        if IMAGE_ONLY:
            return image_embeddings

        styletips_key = self.styletips_embedding(styletips[:, :, 0])
        styletips_value = self.styletips_embedding(styletips[:, :, 1])

        celebrity = self.celebrity_embedding(celebrity)

        attributes_num = attributes.size(1)
        attribute_keys = self.attributes_key_embedding(
            attributes[:, :, 0].view(batch_size, -1)).view(batch_size, attributes_num, -1)
        attribute_values = self.attributes_value_embedding(
            attributes[:, :, 1].view(batch_size, -1)).view(batch_size, attributes_num, -1)

        if EMB_ADD_LAYER_NORM:
            styletips_key = self.style_layer_norm(styletips_key)
            styletips_value = self.style_layer_norm(styletips_value)

            celebrity = self.celebrity_layer_norm(celebrity)

            attribute_keys = self.attribute_key_layer_norm(attribute_keys)
            attribute_values = self.attribute_value_layer_norm(attribute_values)

        style_tips_query_key = self.query_key1(styletips_key)
        style_tips_image_key = self.image_key1(styletips_key)
        style_tips_value = self.value1(styletips_value)

        celebrity_query_key = self.query_key2(celebrity)
        celebrity_image_key = self.image_key2(celebrity)
        celebrity_value = self.value2(celebrity)

        attribute_query_key = self.query_key3(attribute_keys)
        attribute_image_key = self.image_key3(attribute_keys)
        attribute_value = self.value3(attribute_values)

        query_key_ls = []
        image_key_ls = []
        val_ls = []
        if not DISABLE_STYLETIPS:
            query_key_ls.append(style_tips_query_key)
            image_key_ls.append(style_tips_image_key)
            val_ls.append(style_tips_value)
        if not DISABLE_CELEBRITY:
            query_key_ls.append(celebrity_query_key)
            image_key_ls.append(celebrity_image_key)
            val_ls.append(celebrity_value)
        if not DISABLE_ATTRIBUTE:
            query_key_ls.append(attribute_query_key)
            image_key_ls.append(attribute_image_key)
            val_ls.append(attribute_value)

        query_key = torch.cat(query_key_ls, dim=1)
        image_key = torch.cat(image_key_ls, dim=1)
        value = torch.cat(val_ls, dim=1)

        query_aware_alpha = self.attn_dropout(torch.softmax(torch.einsum("bh,bih->bi", queries, query_key), dim=-1))
        image_aware_alpha = self.attn_dropout(
            torch.softmax(torch.einsum("bh,bih->bi", image_embeddings, image_key), dim=-1))

        query_aware_knowledge = self.dropout(self.query_o(torch.einsum("bi,bih->bh", query_aware_alpha, value)))
        image_aware_knowledge = self.dropout(self.image_o(torch.einsum("bi,bih->bh", image_aware_alpha, value)))

        hidden_state = torch.cat([query_aware_knowledge, image_aware_knowledge, image_embeddings], dim=-1)
        hidden_state = self.out_mlp(hidden_state)

        return hidden_state
