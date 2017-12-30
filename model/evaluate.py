import torch
import numpy as np
import itertools

import model.utils as utils


def calc_score(ner_model, dataset_loader, if_cuda):

    ner_model.eval()
    correct = 0
    total_act = 0
    for feature, label, action in itertools.chain.from_iterable(dataset_loader):  # feature : torch.Size([4, 17])
        fea_v, tg_v, ac_v = utils.repack_vb(if_cuda, feature, label, action)
        loss, pre_action = ner_model.forward(fea_v, ac_v)  # loss torch.Size([1, seq_len, action_size+1, action_size+1])
        for idx in range(len(pre_action)):
            if pre_action[idx] == ac_v.squeeze(0).data[idx]:
                correct += 1
        total_act += len(pre_action)

    acc = correct / float(total_act)

    return acc

def calc_f1_score(ner_model, dataset_loader, action2idx, if_cuda):

    idx2action = {v: k for k, v in action2idx.items()}
    ner_model.eval()
    correct = 0
    total_correct_entity = 0
    total_act = 0

    total_entity_in_gold = 0
    total_entity_in_pre = 0
    for feature, label, action in itertools.chain.from_iterable(dataset_loader):  
        fea_v, tg_v, ac_v = utils.repack_vb(if_cuda, feature, label, action)
        loss, pre_action, right_num = ner_model.forward(fea_v, ac_v)

        num_entity_in_real, num_entity_in_pre, correct_entity = to_entity(ac_v.squeeze(0).data.tolist(), pre_action, idx2action)
        for idx in range(len(pre_action)):
            if pre_action[idx] == ac_v.squeeze(0).data[idx]:
                correct += 1
        total_act += len(pre_action)
        total_correct_entity += correct_entity
        total_entity_in_gold += num_entity_in_real
        total_entity_in_pre += num_entity_in_pre

    acc = correct / float(total_act)
    if total_entity_in_pre > 0 :
        pre = total_correct_entity / float(total_entity_in_pre)
    else:
        pre = 0
    if total_entity_in_gold > 0 :
        rec = total_correct_entity / float(total_entity_in_gold)
    else:
        rec = 0
    if (pre + rec) > 0:
        f1 = 2 * pre * rec / float(pre + rec)
    else:
        f1 = 0
    return f1, pre, rec, acc

def to_entity(real_action, predict_action, idx2action):
    flags = [False, False]
    entitys = [[],[]]
    actions = [real_action, predict_action]
    for idx in range(len(actions)):
        ner_start_pos = -1
        for ac_idx in range(len(actions[idx])):
            if idx2action[actions[idx][ac_idx]].startswith('S') and ner_start_pos < 0:
                ner_start_pos = ac_idx
            elif idx2action[actions[idx][ac_idx]].startswith('O') and ner_start_pos >= 0:
                ner_start_pos = -1
            elif idx2action[actions[idx][ac_idx]].startswith('R') and ner_start_pos >= 0:
                entitys[idx].append(str(ner_start_pos)+'-'+str(ac_idx-1)+idx2action[actions[idx][ac_idx]])
                ner_start_pos = -1
    correct_entity = set(entitys[0]) & set(entitys[1])
    return len(entitys[0]), len(entitys[1]), len(correct_entity)

