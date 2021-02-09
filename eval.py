import numpy as np

import eval_metric as evall


def evalll(r):
    p_5 = 0
    p_10 = 0
    p_20 = 0

    recall_5 = 0
    recall_10 = 0
    recall_20 = 0

    ndcg_5 = 0
    ndcg_10 = 0
    ndcg_20 = 0

    for i in r:

        p_5 = p_5 + evall.precision_at_k(i, 5)
        p_10 = p_10 + evall.precision_at_k(i, 10)
        p_20 = p_20 + evall.precision_at_k(i, 20)

        recall_5 = recall_5 + evall.recall_at_k(i, 5)
        recall_10 = recall_10 + evall.recall_at_k(i, 10)
        recall_20 = recall_20 + evall.recall_at_k(i, 20)

        ndcg_5 = ndcg_5 + evall.ndcg_at_k(i, 5)
        ndcg_10 = ndcg_10 + evall.ndcg_at_k(i, 10)
        ndcg_20 = ndcg_20 + evall.ndcg_at_k(i, 20)

    p_5 = p_5/len(r)
    p_10 = p_10/len(r)
    p_20 = p_20 / len(r)

    recall_5 = recall_5 / len(r)
    recall_10 = recall_10 / len(r)
    recall_20 = recall_20 / len(r)

    ndcg_5 = ndcg_5 / len(r)
    ndcg_10 = ndcg_10 / len(r)
    ndcg_20 = ndcg_20 / len(r)

    # print('--p_5:', str(p_5), '--p_10:', str(p_10), '--p_20:', str(p_20),
    #       '--recall_5:', str(recall_5), '--recall_10:', str(recall_10), '--recall_20:', str(recall_20),
    #       '--ndcg_5:', str(ndcg_5), '--ndcg_10:', str(ndcg_10), '--ndcg_20:', str(ndcg_20))

    return p_5, p_10, p_20, recall_5, recall_10, recall_20, ndcg_5, ndcg_10, ndcg_20

def ranklist_to_binarylist(ranklist, pos_image_list):
    binarylist = []
    for i in range(len(ranklist)):
        rank = ranklist[i]
        rank = np.array(rank)
        rank = np.argsort(-rank)
        for j in range(len(rank)):
            if rank[j] in pos_image_list[i]:
                rank[j] = 1
            else:
                rank[j] = 0
        binarylist.append(rank)
    return binarylist

def eval(ranklist, pos_image_list):
    '''
    :param ranklist:     [num_of_query, 1000]   query的1000个images的score （）
    :param pos_image_list: [num_of_query, ? ]   query的 正例image的 位置，
    比如 ：
      第0个query的images 的第 0个，第2个是正例
      第1个query的images 的第 2个，第3个，第4个是正例
    [
      [0,2],
      [2,3,4]
    ]
    :return:
    '''
    binarylist = ranklist_to_binarylist(ranklist, pos_image_list)
    return evalll(binarylist)
