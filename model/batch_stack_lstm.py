import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

import model.utils as utils


class TransitionNER(nn.Module):
    def __init__(self, mode, action2idx, word2idx, label2idx, char2idx, ner_map, vocab_size, action_size, embedding_dim,
                 action_embedding_dim, char_embedding_dim,
                 hidden_dim, char_hidden_dim, rnn_layers, dropout_ratio, use_spelling, char_structure, is_cuda):
        super(TransitionNER, self).__init__()
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.action2idx = action2idx
        self.label2idx = label2idx
        self.char2idx = char2idx
        self.use_spelling = use_spelling
        self.char_structure = char_structure
        if is_cuda >= 0:
            self.gpu_triger = True
        else:
            self.gpu_triger = False
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.idx2action = {v: k for k, v in action2idx.items()}
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.ner_map = ner_map

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.action_embeds = nn.Embedding(action_size, action_embedding_dim)
        self.relation_embeds = nn.Embedding(action_size, action_embedding_dim)

        if self.use_spelling:
            self.char_embeds = nn.Embedding(len(self.char2idx), char_embedding_dim)
            if self.char_structure == 'lstm':
                self.tok_embedding_dim = self.embedding_dim + char_hidden_dim * 2
                self.unk_char_embeds = nn.Parameter(torch.randn(1, char_hidden_dim * 2), requires_grad=True)
                self.pad_char_embeds = nn.Parameter(torch.zeros(1, char_hidden_dim * 2))
                self.char_bi_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=rnn_layers,
                                            bidirectional=True, dropout=dropout_ratio)
            elif self.char_structure == 'cnn':
                self.tok_embedding_dim = self.embedding_dim + char_hidden_dim
                self.pad_char_embeds = nn.Parameter(torch.zeros(1, char_hidden_dim))
                self.unk_char_embeds = nn.Parameter(torch.randn(1, char_hidden_dim), requires_grad=True)
                self.conv1d = nn.Conv1d(char_embedding_dim, char_hidden_dim, 3, padding=2)
        else:
            self.tok_embedding_dim = self.embedding_dim

        self.buffer_lstm = nn.LSTMCell(self.tok_embedding_dim, hidden_dim)
        self.stack_lstm = nn.LSTMCell(self.tok_embedding_dim, hidden_dim)
        self.action_lstm = nn.LSTMCell(action_embedding_dim, hidden_dim)
        self.output_lstm = nn.LSTMCell(self.tok_embedding_dim, hidden_dim)
        self.entity_forward_lstm = nn.LSTMCell(self.tok_embedding_dim, hidden_dim)
        self.entity_backward_lstm = nn.LSTMCell(self.tok_embedding_dim, hidden_dim)

        self.ac_lstm = nn.LSTM(action_embedding_dim, hidden_dim, num_layers=rnn_layers, bidirectional=False,
                               dropout=dropout_ratio)
        self.lstm = nn.LSTM(self.tok_embedding_dim, hidden_dim, num_layers=rnn_layers, bidirectional=False,
                            dropout=dropout_ratio)
        self.rnn_layers = rnn_layers

        self.dropout_e = nn.Dropout(p=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.init_buffer = utils.xavier_init(self.gpu_triger, 1, hidden_dim)
        self.empty_emb = nn.Parameter(torch.randn(1, hidden_dim))
        self.lstm_padding = nn.Parameter(torch.randn(1, self.tok_embedding_dim))
        self.lstms_output_2_softmax = nn.Linear(hidden_dim * 4, hidden_dim)
        self.output_2_act = nn.Linear(hidden_dim, len(ner_map) + 2)
        self.entity_2_output = nn.Linear(hidden_dim * 2 + action_embedding_dim, self.tok_embedding_dim)

        self.batch_size = 1
        self.seq_length = 1

    def _rnn_get_output(self, state):
        return state[0]

    def get_possible_actions(self, stack, buffer):
        valid_actions = []
        if len(buffer) > 0:
            valid_actions.append(self.action2idx["SHIFT"])
        if len(stack) > 0:
            valid_actions += [self.action2idx[ner_action] for ner_action in self.ner_map.keys()]
        else:
            valid_actions.append(self.action2idx["OUT"])
        return valid_actions

    def get_possible_actions_batch(self, stacks, buffer_lens, have_action_batch):
        assert len(stacks) == len(buffer_lens)
        valid_actions = [[] for i in range(len(buffer_lens))]
        for i in have_action_batch:
            if buffer_lens[i] > 0:
                valid_actions[i].append(self.action2idx["SHIFT"])
            if stacks[i][1] != '<pad>':
                valid_actions[i] += [self.action2idx[ner_action] for ner_action in self.ner_map.keys()]
            else:
                valid_actions[i].append(self.action2idx["OUT"])

        return valid_actions

    def getloss_batch(self, have_action_batch, batch_buffer, batch_stack, batch_action, batch_output,
                      batch_valid_actions, batch_real_actions=None):
        predict_actions = []
        losses = []
        if self.mode == 'train':
            lstms_output = [torch.cat(
                [batch_buffer[batch_idx][0], batch_stack[batch_idx][0][0], batch_output[batch_idx][0][0],
                 batch_action[batch_idx]], 1)
                            for batch_idx in have_action_batch]
        elif self.mode == 'predict':
            lstms_output = [torch.cat(
                [batch_buffer[batch_idx][0], batch_stack[batch_idx][0][0], batch_output[batch_idx][0][0],
                 batch_action[batch_idx][0][0]], 1)
                            for batch_idx in have_action_batch]
        lstms_output = torch.cat([i for i in lstms_output], 0)
        hidden_output = torch.tanh(self.lstms_output_2_softmax(self.dropout(lstms_output)))
        logits = self.output_2_act(hidden_output)
        for idx in range(len(have_action_batch)):
            logit = logits[idx][
                utils.variable(torch.LongTensor(batch_valid_actions[have_action_batch[idx]]), self.gpu_triger)]
            valid_action_tbl = {a: i for i, a in enumerate(batch_valid_actions[have_action_batch[idx]])}
            log_probs = torch.nn.functional.log_softmax(logit)
            action_idx = torch.max(log_probs.cpu(), 0)[1][0].data.numpy()[0]
            action_predict = batch_valid_actions[have_action_batch[idx]][action_idx]
            predict_actions.append(action_predict)
            if self.mode == 'train':
                if log_probs is not None:
                    losses.append(log_probs[valid_action_tbl[batch_real_actions[have_action_batch[idx]]]])

        if self.mode == 'predict':
            losses = None

        return predict_actions, losses

    def rand_init_hidden(self):

        if self.gpu_triger is True:
            return autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)).cuda(), autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)).cuda()
        else:
            return autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_seq_size(self, sentence):

        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = 1

    def set_batch_seq_size(self, sentence):

        tmp = sentence.size()
        self.seq_length = tmp[1]
        self.batch_size = tmp[0]

    def load_pretrained_embedding(self, pre_embeddings):

        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def rand_init(self, init_word_embedding=False, init_action_embedding=True, init_relation_embedding=True):

        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if init_action_embedding:
            utils.init_embedding(self.action_embeds.weight)
        if init_relation_embedding:
            utils.init_embedding(self.relation_embeds.weight)

        if self.use_spelling:
            utils.init_embedding(self.char_embeds.weight)
        if self.use_spelling and self.char_structure == 'lstm':
            utils.init_lstm(self.char_bi_lstm)

        utils.init_linear(self.lstms_output_2_softmax)
        utils.init_linear(self.output_2_act)
        utils.init_linear(self.entity_2_output)

        utils.init_lstm(self.lstm)
        utils.init_lstm_cell(self.buffer_lstm)
        utils.init_lstm_cell(self.action_lstm)
        utils.init_lstm_cell(self.stack_lstm)
        utils.init_lstm_cell(self.output_lstm)
        utils.init_lstm_cell(self.entity_forward_lstm)
        utils.init_lstm_cell(self.entity_backward_lstm)

    def batch_shift_out(self, Action, buffer, stack, batch_shift_idx):
        from_buffer_2_stack = [buffer[i].pop() for i in batch_shift_idx]
        lstm_in = torch.cat([i[1] for i in from_buffer_2_stack], 0)
        lstm_h = torch.cat([stack[i][-1][0][0] for i in batch_shift_idx], 0)
        lstm_c = torch.cat([stack[i][-1][0][1] for i in batch_shift_idx], 0)

        if Action == 'S':
            h, c = self.stack_lstm(lstm_in, (lstm_h, lstm_c))
        elif Action == 'O':
            h, c = self.output_lstm(lstm_in, (lstm_h, lstm_c))

        i = 0
        for id in batch_shift_idx:
            stack[id].append(
                [(h[i].unsqueeze(0), c[i].unsqueeze(0)), from_buffer_2_stack[i][2], from_buffer_2_stack[i][1]])
            i += 1

        return buffer, stack

    def batch_reduce(self, stack, output, batch_relation, batch_reduce_idx):
        output_input = []
        for idx in batch_reduce_idx:
            entity = []
            ent = ''
            (ent_f_h, ent_f_c) = (utils.xavier_init(self.gpu_triger, 1, self.hidden_dim),
                                  utils.xavier_init(self.gpu_triger, 1, self.hidden_dim))
            (ent_b_h, ent_b_c) = (utils.xavier_init(self.gpu_triger, 1, self.hidden_dim),
                                  utils.xavier_init(self.gpu_triger, 1, self.hidden_dim))

            while stack[idx][-1][1] != '<pad>':
                _, word, tok_emb = stack[idx].pop()
                entity.append([tok_emb, word])
            for ent_idx in range(len(entity)):
                ent = ent + ' ' + word
                ent_f_h, ent_f_c = self.entity_forward_lstm(entity[ent_idx][0], (ent_f_h, ent_f_c))
                ent_b_h, ent_b_c = self.entity_backward_lstm(entity[len(entity) - ent_idx - 1][0], (ent_b_h, ent_b_c))
            entity_input = self.dropout(torch.cat([ent_b_h, ent_f_h], 1))
            output_input.append([self.entity_2_output(torch.cat([entity_input, batch_relation[idx]], 1)), ent])

        lstm_in = torch.cat([ent_emb[0] for ent_emb in output_input])
        lstm_h = torch.cat([output[i][-1][0][0] for i in batch_reduce_idx], 0)
        lstm_c = torch.cat([output[i][-1][0][1] for i in batch_reduce_idx], 0)
        h, c = self.output_lstm(lstm_in, (lstm_h, lstm_c))
        h = self.dropout(h)
        i = 0
        for id in batch_reduce_idx:
            output[id].append([(h[i].unsqueeze(0), c[i].unsqueeze(0)), output_input[i][1], output_input[i][0]])
            i += 1

        return stack, output

    def forward(self, sentences, actions=None, hidden=None):

        if actions is not None:
            self.mode = "train"
        else:
            self.mode = "predict"

        self.set_batch_seq_size(sentences)  # sentences [batch_size, max_len]
        word_embeds = self.dropout_e(self.word_embeds(sentences))  # [batch_size, max_len, embeddind_size]
        if self.mode == 'train':
            action_embeds = self.dropout_e(self.action_embeds(actions))
            relation_embeds = self.dropout_e(self.relation_embeds(actions))
            action_output, _ = self.ac_lstm(action_embeds.transpose(0, 1))
            action_output = action_output.transpose(0, 1)

        lstm_initial = (
        utils.xavier_init(self.gpu_triger, 1, self.hidden_dim), utils.xavier_init(self.gpu_triger, 1, self.hidden_dim))

        sentence_array = sentences.data.cpu().numpy()
        sents_len = []
        token_embedds = None
        for sent_idx in range(len(sentence_array)):
            count_words = 0
            token_embedding = None
            for word_idx in reversed(range(len(sentence_array[sent_idx]))):
                if self.use_spelling:
                    if sentence_array[sent_idx][word_idx] == 1:
                        tok_rep = torch.cat([word_embeds[sent_idx][word_idx].unsqueeze(0), self.pad_char_embeds], 1)
                    elif sentence_array[sent_idx][word_idx] == 0:
                        count_words += 1
                        tok_rep = torch.cat([word_embeds[sent_idx][word_idx].unsqueeze(0), self.unk_char_embeds], 1)
                    else:
                        count_words += 1
                        word = sentence_array[sent_idx][word_idx]
                        chars_in_word = [self.char2idx[char] for char in self.idx2word[word]]
                        chars_Tensor = utils.variable(torch.from_numpy(np.array(chars_in_word)), self.gpu_triger)
                        chars_embeds = self.dropout_e(self.char_embeds(chars_Tensor))
                        if self.char_structure == 'lstm':
                            char_o, hidden = self.char_bi_lstm(chars_embeds.unsqueeze(1), hidden)
                            char_out = torch.chunk(hidden[0].squeeze(1), 2, 0)
                            tok_rep = torch.cat(
                                [word_embeds[sent_idx][word_idx].unsqueeze(0), char_out[0], char_out[1]], 1)
                        elif self.char_structure == 'cnn':
                            char, _ = self.conv1d(chars_embeds.unsqueeze(0).transpose(1, 2)).max(
                                dim=2)  # [batch_size, Embedding_sie, sentence_len] --> [batch_size, output_dim, sentence_len+padding_num*2 - kernel_num + 1]
                            char = torch.tanh(char)
                            tok_rep = torch.cat([word_embeds[sent_idx][word_idx].unsqueeze(0), char], 1)
                else:
                    if sentence_array[sent_idx][word_idx] != 1:
                        count_words += 1
                    tok_rep = word_embeds[sent_idx][word_idx].unsqueeze(0)
                if token_embedding is None:
                    token_embedding = tok_rep
                else:
                    token_embedding = torch.cat([token_embedding, tok_rep], 0)

            sents_len.append(count_words)
            if token_embedds is None:
                token_embedds = token_embedding.unsqueeze(0)
            else:
                token_embedds = torch.cat([token_embedds, token_embedding.unsqueeze(0)], 0)

        tokens = token_embedds.transpose(0, 1)
        tok_output, hidden = self.lstm(tokens)  # [max_len, batch_size, hidden_dim]
        tok_output = tok_output.transpose(0, 1)

        buffer = [[] for i in range(self.batch_size)]
        losses = [[] for i in range(self.batch_size)]
        right = [0 for i in range(self.batch_size)]
        predict_actions = [[] for i in range(self.batch_size)]
        output = [[[lstm_initial, "<pad>"]] for i in range(self.batch_size)]
        if self.mode == 'predict':
            action = [[[lstm_initial, "<pad>"]] for i in range(self.batch_size)]

        for idx in range(tok_output.size(0)):
            for word_idx in range(tok_output.size(1)):
                buffer[idx].append([tok_output[idx][word_idx].unsqueeze(0), token_embedds[idx][word_idx].unsqueeze(0),
                                    self.idx2word[sentence_array[idx][tok_output.size(1) - 1 - word_idx]]])

        stack = [[[lstm_initial, "<pad>"]] for i in range(self.batch_size)]
        for act_idx in range(self.seq_length):
            batch_buffer = [b[-1] for b in buffer]
            if self.mode == 'train':
                if act_idx == 0:
                    batch_action = [lstm_initial[0] for a in range(self.batch_size)]
                else:
                    batch_action = [a[act_idx - 1].unsqueeze(0) for a in action_output]
                batch_relation = [r[act_idx].unsqueeze(0) for r in relation_embeds]
            elif self.mode == 'predict':
                batch_action = [a[-1] for a in action]
            batch_output = [o[-1] for o in output]
            batch_stack = [s[-1] for s in stack]

            have_action_batch_1 = [i for i in range(len(sents_len)) if sents_len[i] > 0]
            have_action_batch_2 = [i for i in range(len(batch_stack)) if batch_stack[i][1] != '<pad>']
            have_action_batch = list(set(have_action_batch_1).union(set(have_action_batch_2)))

            if len(have_action_batch) > 0:
                batch_valid_actions = self.get_possible_actions_batch(batch_stack, sents_len, have_action_batch)
                if self.mode == 'train':
                    batch_real_action = [ac[act_idx] for ac in actions.data]
                    batch_pred, batch_loss = self.getloss_batch(have_action_batch, batch_buffer, batch_stack,
                                                                batch_action, batch_output, batch_valid_actions,
                                                                batch_real_action)
                    batch_real_action = [self.idx2action[ac] for ac in batch_real_action]
                elif self.mode == 'predict':
                    batch_pred, batch_loss = self.getloss_batch(have_action_batch, batch_buffer, batch_stack,
                                                                batch_action, batch_output, batch_valid_actions)
                    pred_action_tensor = utils.variable(torch.from_numpy(np.array(batch_pred)), self.gpu_triger)
                    predict_actions_embed = self.dropout_e(self.action_embeds(pred_action_tensor))
                    ac_lstm_h, ac_lstm_c = self.action_lstm(predict_actions_embed, (torch.cat(
                        [action[ac_idx][-1][0][0] for ac_idx in range(len(action)) if ac_idx in have_action_batch]),
                                                                                    torch.cat(
                                                                                        [action[ac_idx][-1][0][1] for
                                                                                         ac_idx in range(len(action)) if
                                                                                         ac_idx in have_action_batch])))

                i = 0
                for batch_idx in range(self.batch_size):
                    if batch_idx in have_action_batch:
                        predict_actions[batch_idx].append(batch_pred[i])
                        if self.mode == 'train':
                            losses[batch_idx].append(batch_loss[i])
                        elif self.mode == 'predict':
                            action[batch_idx].append([(ac_lstm_h[i].unsqueeze(0), ac_lstm_c[i].unsqueeze(0)),
                                                      self.idx2action[batch_pred[i]]])
                        i += 1
                    else:
                        if self.mode == 'predict':
                            action[batch_idx].append([lstm_initial, "<pad>"])

                if self.mode == 'predict':
                    batch_real_action = [ac[-1][1] for ac in action]
                    relation_embeds = self.dropout_e(self.relation_embeds(
                        utils.variable(torch.from_numpy(np.array([self.action2idx[a] for a in batch_real_action])),
                                      self.gpu_triger)))
                    batch_relation = [relation_embed.unsqueeze(0) for relation_embed in relation_embeds]

                batch_shift_idx = [idx for idx in range(len(batch_real_action)) if
                                   batch_real_action[idx].startswith('S')]
                batch_out_idx = [idx for idx in range(len(batch_real_action)) if batch_real_action[idx].startswith('O')]
                batch_reduce_idx = [idx for idx in range(len(batch_real_action)) if
                                    batch_real_action[idx].startswith('R')]

                # batch_relation = [batch_relation[i] for i in batch_reduce_idx]
                if len(batch_shift_idx) > 0:
                    buffer, stack = self.batch_shift_out('S', buffer, stack, batch_shift_idx)
                    for i in range(len(sents_len)):
                        if i in batch_shift_idx:
                            sents_len[i] -= 1
                if len(batch_out_idx) > 0:
                    buffer, output = self.batch_shift_out('O', buffer, output, batch_out_idx)
                    for i in range(len(sents_len)):
                        if i in batch_out_idx:
                            sents_len[i] -= 1
                if len(batch_reduce_idx) > 0:
                    stack, output = self.batch_reduce(stack, output, batch_relation, batch_reduce_idx)
        loss = 0
        if self.mode == 'train':
            for idx in range(self.batch_size):
                loss += -torch.sum(torch.cat(losses[idx]))

        return loss, predict_actions
