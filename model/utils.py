import itertools
import json

import numpy as np
import torch.nn as nn
import torch.nn.init
from torch.utils.data import Dataset

class TransitionDataset_P(Dataset):

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return len(self.data_tensor)

class TransitionDataset(Dataset):

    def __init__(self, data_tensor, label_tensor, action_tensor):

        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.action_tensor = action_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index], self.action_tensor[index]

    def __len__(self):
        return len(self.data_tensor)


zip = getattr(itertools, 'izip', zip)

def varible(tensor, gpu):
    if gpu:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


def xavier_init(gpu, *size):
    return nn.init.xavier_normal(varible(torch.FloatTensor(*size), gpu))


def init_varaible_zero(gpu, *size):
    return varible(torch.zeros(*size),gpu)

def to_scalar(var):

    return var.view(-1).data.tolist()[0]


def argmax(vec):

    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, m_size):

    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
      
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def encode2char_safe(input_lines, char_dict):

    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def encode_safe(input_lines, word_dict, unk, singleton, singleton_rate):
    lines = list()
    for sentence in input_lines:
        line = list()
        for word in sentence:
            if word in singleton and torch.rand(1).numpy()[0] < singleton_rate:
                line.append(unk)
            elif word in word_dict:
                line.append(word_dict[word])
            else:
                line.append(unk)
        lines.append(line)
    return lines

def encode_safe_predict(input_lines, word_dict, unk):
    lines = list(map(lambda t: list(map(lambda m: word_dict.get(m, unk), t)), input_lines))
    return lines

def encode(input_lines, word_dict):

    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines

def shrink_features(feature_map, features, thresholds):

    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    #inserting unk to be 0 encoded
    feature_map['<unk>'] = 0
    #inserting eof
    feature_map['<eof>'] = len(feature_map)
    return feature_map

def generate_corpus(lines: object, word_count, if_shrink_feature: object = False, thresholds: object = 1) -> object:

    feature_map = dict()
    char_map = {"<start>": 0, "<end>": 1}
    label_map = dict()
    action_map = {"OUT": 0, "SHIFT": 1}
    ner_map =dict()

    features = list()
    actions = list()
    labels = list()

    tmp_fl = list()
    tmp_ll = list()
    tmp_al = list()
    count_ner = 0
    ner_label = ""
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] in word_count:
                word_count[line[0]] += 1
            else:
                word_count[line[0]] = 1
            for char_idx in range(len(line[0])):
                if line[0][char_idx] not in char_map:
                    char_map[line[0][char_idx]] = len(char_map)
            tmp_ll.append(line[-1])

            if line[0] not in feature_map:
                feature_map[line[0]] = len(feature_map) + 1 #0 is for unk
            if line[-1] not in label_map:
                label_map[line[-1]] = len(label_map)

            if len(line[-1].split('-')) > 1:
                if line[-1].split('-')[0] == "B" and not ner_label == "":
                    tmp_al.append(ner_label)
                    count_ner += 1
                ner_label = "REDUCE-"+line[-1].split('-')[1]
                if ner_label not in action_map:
                    ner_map[ner_label] = len(ner_map)
                    action_map[ner_label] = len(action_map)
                tmp_al.append("SHIFT")
            else:
                if not ner_label == "":
                    tmp_al.append(ner_label)
                    count_ner += 1
                    ner_label = ""
                tmp_al.append("OUT")

        elif len(tmp_fl) > 0:
            if not ner_label =="":
                tmp_al.append(ner_label)
                count_ner += 1
                ner_label = ""
            assert len(tmp_ll) == len(tmp_fl)
            assert len(tmp_al) == len(tmp_fl)+count_ner
            features.append(tmp_fl)
            labels.append(tmp_ll)
            actions.append(tmp_al)
            count_ner = 0
            tmp_al = list()
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        assert len(tmp_ll) == len(tmp_fl)
        assert len(tmp_al) == len(tmp_fl)+count_ner
        features.append(tmp_fl)
        labels.append(tmp_ll)
        actions.append(tmp_al)

    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        #inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        #inserting eof
        feature_map['<eof>'] = len(feature_map)

    singleton = list()
    for k, v in word_count.items():
        if v == 1:
            singleton.append(k)

    return features, labels, actions, feature_map, label_map, action_map, char_map, ner_map, singleton


def read_corpus_ner(lines, word_count):

    features = list()
    actions = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    tmp_al = list()
    count_ner = 0
    ner_label = ""
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            if line[0] in word_count:
                word_count[line[0]] += 1
            else:
                word_count[line[0]] = 1
            tmp_ll.append(line[-1])
            if len(line[-1].split('-')) > 1:
                if line[-1].split('-')[0] == "B" and not ner_label == "":
                    tmp_al.append(ner_label)
                    count_ner += 1
                ner_label = "REDUCE-"+line[-1].split('-')[1]
                tmp_al.append("SHIFT")
            else:
                if not ner_label == "":
                    tmp_al.append(ner_label)
                    count_ner += 1
                    ner_label = ""
                tmp_al.append("OUT")

        elif len(tmp_fl) > 0:
            if not ner_label =="":
                tmp_al.append(ner_label)
                count_ner += 1
                ner_label = ""
            assert len(tmp_ll) == len(tmp_fl)
            assert len(tmp_al) == len(tmp_fl)+count_ner
            features.append(tmp_fl)
            labels.append(tmp_ll)
            actions.append(tmp_al)
            count_ner = 0
            tmp_al = list()
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        assert len(tmp_ll) == len(tmp_fl)
        assert len(tmp_al) == len(tmp_fl)+count_ner
        features.append(tmp_fl)
        labels.append(tmp_ll)
        actions.append(tmp_al)

    return features, labels, actions, word_count

def read_corpus_predict(lines):
    features = list()
    for line in lines:
        line = line.rstrip('\n').split()
        features.append(line)

    return features



def shrink_embedding(feature_map, word_dict, word_embedding, caseless):

    if caseless:
        feature_map = set([k.lower() for k in feature_map.keys()])
    new_word_list = [k for k in word_dict.keys() if (k in feature_map)]
    new_word_dict = {k:v for (v, k) in enumerate(new_word_list)}
    new_word_list_ind = torch.LongTensor([word_dict[k] for k in new_word_list])
    new_embedding = word_embedding[new_word_list_ind]
    return new_word_dict, new_embedding

def load_embedding_wlm(emb_file, delimiter, feature_map, full_feature_set, caseless, unk, emb_len, shrink_to_train=False, shrink_to_corpus=False):

    if caseless:
        feature_set = set([key.lower() for key in feature_map])
        full_feature_set = set([key.lower() for key in full_feature_set])
    else:
        feature_set = set([key for key in feature_map])
        full_feature_set = set([key for key in full_feature_set])

    # ensure <unk> is 0
    word_dict = {v: (k + 1) for (k, v) in enumerate(feature_set - set(['<unk>']))}
    word_dict['<unk>'] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        if len(line) > 2:
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

            if shrink_to_train and line[0] not in feature_set:
                continue

            if line[0] == unk:
                rand_embedding_tensor[0] = torch.FloatTensor(vector)  # unk is 0
            elif line[0] in word_dict:
                rand_embedding_tensor[word_dict[line[0]]] = torch.FloatTensor(vector)
            elif line[0] in full_feature_set:
                indoc_embedding_array.append(vector)
                indoc_word_array.append(line[0])
            elif not shrink_to_corpus:
                outdoc_word_array.append(line[0])
                outdoc_embedding_array.append(vector)

    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        word_emb_len = embedding_tensor_0.size(1)
        assert (word_emb_len == emb_len)

    if shrink_to_corpus:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)

    return word_dict, embedding_tensor



def construct_dataset(input_features, input_label, input_action, word_dict, label_dict, action_dict, singleton, singleton_rate, caseless):

    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x, t)), input_features))
    features = encode_safe(input_features, word_dict, word_dict['<unk>'], singleton, singleton_rate)
    labels = encode(input_label, label_dict)
    actions = encode(input_action, action_dict)
    feature_tensor = []
    label_tensor = []
    action_tensor =[]
    for feature, label, action in zip(features, labels, actions):
        feature_tensor.append(torch.LongTensor(feature))
        label_tensor.append(torch.LongTensor(label))
        action_tensor.append(torch.LongTensor(action))

    dataset = TransitionDataset(feature_tensor, label_tensor, action_tensor)

    return dataset

def construct_dataset_predict(input_features, word_dict, caseless):
    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x, t)), input_features))
    features = encode_safe_predict(input_features, word_dict, word_dict['<unk>'])
    feature_tensor = []
    for feature in features:
        feature_tensor.append(torch.LongTensor(feature))
    dataset = TransitionDataset_P(feature_tensor)

    return dataset


def save_checkpoint(state, track_list, filename):

    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def adjust_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_embedding(input_embedding):

    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):

    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_lstm(input_lstm):

    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        bias = np.sqrt(6.0 / (weight.size(0)/4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_lstm_cell(input_lstm):

    weight = eval('input_lstm.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)
    weight = eval('input_lstm.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        weight = eval('input_lstm.bias_ih' )
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        weight = eval('input_lstm.bias_hh')
        weight.data.zero_()
        weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

def repack_vb(if_cuda, feature, label, action):

    if if_cuda:
        fea_v = torch.autograd.Variable(feature).cuda()  
        label_v = torch.autograd.Variable(label).cuda() 
        action_v = torch.autograd.Variable(action).cuda()  
    else:
        fea_v = torch.autograd.Variable(feature)
        label_v = torch.autograd.Variable(label).contiguous()
        action_v = torch.autograd.Variable(action).contiguous()
    return fea_v, label_v, action_v