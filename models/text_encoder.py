import torch
from PIL import Image
from torch import nn
from torchnlp.word_to_vector.glove import GloVe
from torchvision.models import resnet18
from torchvision.transforms import transforms

from constants import DATA_DIR, DEVICE, DUMP_DIR


class QueryEncoder(nn.Module):
    EMPTY_IMAGE = torch.zeros(3, 64, 64)
    transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # # Version 1.0
    # image_output_size = 300
    # utt_rnn_hidden_size = 256
    # utt_rnn_num_layers = 1
    # dialog_rnn_hidden_size = 256
    # dialog_rnn_num_layers = 1
    # Version 1.1
    image_output_size = 512
    utt_rnn_hidden_size = 256
    utt_rnn_num_layers = 1
    dialog_rnn_hidden_size = 1024
    dialog_rnn_num_layers = 1

    output_size = 512

    def __init__(self, raw_data):
        super().__init__()
        self.raw_data = raw_data
        self.vocab = self.raw_data.dialog_vocab
        self.vocab_size = len(self.vocab)
        self.images = raw_data.images
        # self.word_embedding = nn.Embedding(self.vocab_size, 300).to(DEVICE)
        self.image_fc = nn.Linear(2048, self.image_output_size).to(DEVICE)

        self.utt_rnn = nn.LSTM(300, hidden_size=self.utt_rnn_hidden_size,
                               num_layers=self.utt_rnn_num_layers,
                               batch_first=True, bidirectional=True).to(DEVICE)

        self.dialog_rnn = nn.LSTM(self.utt_rnn_hidden_size * 2 + self.image_output_size,
                                  hidden_size=self.dialog_rnn_hidden_size,
                                  num_layers=self.dialog_rnn_num_layers,
                                  batch_first=True, bidirectional=True).to(DEVICE)

        self.summarize_w1 = nn.Linear(self.utt_rnn_hidden_size * 2, 1).to(DEVICE)
        self.summarize_w2 = nn.Linear(self.dialog_rnn_hidden_size * 2, 1).to(DEVICE)

        self.output = nn.Linear(self.dialog_rnn_hidden_size * 2, self.output_size)

        self.apply(self._init_weights)

        pretrained_embedding = self.get_pretrained_embedding()
        self.word_embedding = nn.Embedding(self.vocab_size, 300).from_pretrained(pretrained_embedding, freeze=False)
        self.resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2]).to(DEVICE)

        # For RNN, sequential dropout should be used instead of following dropout?
        self.dropout = nn.Dropout(p=0.1)

        print(f"image_output_size: {self.image_output_size}")
        print(f"utt_rnn_hidden_size: {self.utt_rnn_hidden_size}")
        print(f"utt_rnn_num_layers: {self.utt_rnn_num_layers}")
        print(f"dialog_rnn_hidden_size: {self.dialog_rnn_hidden_size}")
        print(f"dialog_rnn_num_layers: {self.dialog_rnn_num_layers}")
        # print(f"output_size: {self.output_size}")
        print("Multi-modal hierarchical dialogue encoder is loaded.")

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
        weight = nn.Parameter(torch.randn(self.vocab_size, 300), requires_grad=True)
        weight.data.normal_(mean=0.0, std=0.02)
        pretrained_embedding = GloVe(is_include=lambda x: x in self.vocab, cache=str(DUMP_DIR / '.word_vectors_cache'))
        for i, token in enumerate(self.vocab):
            if token in pretrained_embedding:
                weight[i] = pretrained_embedding[token]
        return weight

    @staticmethod
    def attentive_pooling(x, linear: nn.Linear, x_mask=None):
        scores = linear(x).squeeze(-1)
        if x_mask is not None:
            scores = scores + x_mask * -65500.0
        alpha = torch.softmax(scores, dim=-1)
        y = torch.einsum("bs,bsh->bh", alpha, x)
        return y

    def forward(self, words, images, word_mask):
        """
        words: [batch, context_size, max_text_len]
        word_mask: [batch, context_size, max_text_len]
        images: [batch, context_size, max_image_num]
        """
        batch, context_size, max_utt_len = words.size()
        num_image = images.size(-1)
        words = words.view(batch * context_size, -1)
        word_mask = word_mask.view(batch * context_size, -1)
        images = images.view(batch * context_size, -1)

        word_embedding = self.word_embedding(words)

        image_embedding = []
        for i in range(batch * context_size):
            dialog_image_ls = []
            for j in range(images.size(-1)):
                image_id = images[i, j].item()
                image_path = DATA_DIR / 'images' / self.images[image_id]
                if image_id != 0 and image_path.is_file():
                    try:
                        image = Image.open(image_path).convert("RGB")
                        image = QueryEncoder.transform(image)
                    except OSError:
                        image = QueryEncoder.EMPTY_IMAGE
                else:
                    image = QueryEncoder.EMPTY_IMAGE
                dialog_image_ls.append(image)
            dialog_image_ls = torch.stack(dialog_image_ls, dim=0)
            image_embedding.append(dialog_image_ls)
        image_embedding = torch.stack(image_embedding, dim=0).to(DEVICE)
        image_embedding = image_embedding.view(batch * context_size * num_image, 3, 64, 64)
        # print(image_embedding.size())
        image_embedding = self.resnet(image_embedding).view(batch * context_size * num_image, -1)
        image_embedding = self.image_fc(image_embedding)

        utt_hidden, _ = self.utt_rnn(word_embedding)
        utt_h = self.attentive_pooling(utt_hidden, self.summarize_w1, x_mask=word_mask)

        utt_h = utt_h.reshape(batch, context_size, -1)
        image_embedding = image_embedding.reshape(batch, context_size, -1)
        dialog_rnn_input = torch.cat([utt_h, image_embedding], dim=-1)
        dialog_rnn_output, _ = self.dialog_rnn(dialog_rnn_input)

        dialog_h = self.attentive_pooling(dialog_rnn_output, self.summarize_w2, x_mask=None)
        dialog_h = self.output(dialog_h)

        return dialog_h
