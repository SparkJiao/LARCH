import torch
import torch.nn.functional as F


def encode_image(queries, key_memories, value_memories):
    '''
    queries is a [train_batch_size * F] tensor,
    key_memories is a [train_batch_size * M * F] tensor,
    value_memories is a [train_batch_size * M * F] tensor.
    '''
    M = key_memories.size()[1]  # M is the memory_num
    qs = torch.unsqueeze(queries, dim=-1)  # qs is [train_batch_size * F * 1]
    # batch matrix multiplication
    inner_pro = torch.einsum('bij,bjk->bik', key_memories, qs)  # inner_pro is [train_batch_size * M * 1]
    beta = F.softmax(inner_pro, dim=1).view(-1, 1, M)  # beta is [train_batch_size * 1 * M]
    V = torch.einsum('bij,bjk->bik', beta, value_memories)  # V is [train_batch_size * 1 * F]

    #  image_rep is [train_batch_size, F]
    image_rep = torch.squeeze(V, dim=1)

    return image_rep


def encode_image_(queries, key_memories):
    '''
    queries is a [train_batch_size * F] tensor,
    key_memories is a [train_batch_size * M * F] tensor,
    value_memories is a [train_batch_size * M * F] tensor.
    '''
    M = key_memories.size()[1]  # M is the memory_num
    qs = torch.unsqueeze(queries, dim=-1)  # qs is [train_batch_size * F * 1]
    # batch matrix multiplication
    inner_pro = torch.einsum('bij,bjk->bik', key_memories, qs)  # inner_pro is [train_batch_size * M * 1]
    beta = F.softmax(inner_pro, dim=1).view(-1, 1, M)  # beta is [train_batch_size * 1 * M]
    V = torch.einsum('bij,bjk->bik', beta, key_memories)  # V is [train_batch_size * 1 * F]

    #  image_rep is [train_batch_size, F]
    image_rep = torch.squeeze(V, dim=1)

    return image_rep


def encode_image__(queries, images, key_memories):
    '''
    queries is a [train_batch_size * F] tensor,
    key_memories is a [train_batch_size * M * F] tensor,
    value_memories is a [train_batch_size * M * F] tensor.
    '''
    M = key_memories.size()[1]  # M is the memory_num
    qs = torch.unsqueeze(queries, dim=-1)  # qs is [train_batch_size * F * 1]
    # batch matrix multiplication
    inner_pro = torch.einsum('bij,bjk->bik', key_memories, qs)  # inner_pro is [train_batch_size * M * 1]
    beta = F.softmax(inner_pro, dim=1).view(-1, 1, M)  # beta is [train_batch_size * 1 * M]
    V = torch.einsum('bij,bjk->bik', beta, key_memories)  # V is [train_batch_size * 1 * F]

    #  image_rep is [train_batch_size, F]
    knowledge_rep = torch.squeeze(V, dim=1)

    out = torch.cat((images, knowledge_rep), dim=1)
    return out
