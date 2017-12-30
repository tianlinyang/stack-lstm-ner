import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

import model.utils as utils


class StackRNN(object):
    def __init__(self, cell, gpu_triger, dropout, get_output, p_empty_embedding=None):
        self.cell = cell
        self.dropout = dropout
        initial_state = (utils.xavier_init(gpu_triger, 1, cell.hidden_size), utils.xavier_init(gpu_triger, 1, cell.hidden_size))
        self.s = [(initial_state, None)]
        self.empty = None
        self.get_output = get_output
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def push(self, expr, extra=None):
        self.dropout(self.s[-1][0][0])
        self.s.append((self.cell(expr, self.s[-1][0]), extra))

    def pop(self):
        return self.s.pop()[1]

    def embedding(self):
        return self.get_output(self.s[-1][0]) if len(self.s) > 1 else self.empty

    def clear(self):
        self.s.reverse()
        self.back_to_init()

    def back_to_init(self):
        while self.__len__() > 0:
            self.pop()

    def __len__(self):
        return len(self.s) - 1


class TransitionNER(nn.Module):

    def __init__(self, action2idx, word2idx, label2idx, char2idx, ner_map, vocab_size, action_size, embedding_dim, action_embedding_dim, char_embedding_dim,
                 hidden_dim, char_hidden_dim, rnn_layers, dropout_ratio, singleton, singleton_rate, use_spelling, char_structure, is_cuda):
        super(TransitionNER, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.action2idx = action2idx
        self.char2idx = char2idx
        self.label2idx = label2idx
        self.rnn_layers = rnn_layers
        self.singleton = singleton
        self.singleton_rate = singleton_rate
        self.use_spelling = use_spelling
        self.char_structure = char_structure
        if is_cuda >=0:
            self.gpu_triger = True
        else:
            self.gpu_triger = False
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.idx2action = {v: k for k, v in action2idx.items()}
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.ner_map = ner_map

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeds = nn.Embedding(len(self.char2idx), char_embedding_dim)
        self.action_embeds = nn.Embedding(action_size, action_embedding_dim)
        self.relation_embeds = nn.Embedding(action_size, action_embedding_dim)

        if use_spelling and self.char_structure == 'lstm':
            self.tok_embedding_dim = self.embedding_dim + char_hidden_dim*2
            self.unk_char_embeds = nn.Parameter(torch.randn(1, char_hidden_dim * 2), requires_grad=True)
            self.char_bi_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio)
        elif use_spelling and self.char_structure == 'cnn':
            self.tok_embedding_dim = self.embedding_dim + char_hidden_dim
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
        self.rnn_layers = rnn_layers

        self.dropout_e = nn.Dropout(p=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self.empty_emb = nn.Parameter(torch.randn(1, hidden_dim))

        self.lstms_output_2_softmax = nn.Linear(hidden_dim * 4, hidden_dim)
        self.output_2_act = nn.Linear(hidden_dim, len(ner_map)+2)
        self.entity_2_output = nn.Linear(hidden_dim*2 + action_embedding_dim, self.tok_embedding_dim)
        self.buffer_2_output = nn.Linear(self.tok_embedding_dim + action_embedding_dim, self.tok_embedding_dim)

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

    def rand_init_hidden(self):
        """
        random initialize hidden variable
        """
        if self.gpu_triger is True:
            return autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)).cuda(), autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)).cuda()
        else:
            return autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
                torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = 1

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def rand_init(self, init_word_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        utils.init_embedding(self.action_embeds.weight)
        utils.init_embedding(self.relation_embeds.weight)
        utils.init_embedding(self.char_embeds.weight)

        if self.use_spelling and self.char_structure == 'lstm':
            utils.init_lstm(self.char_bi_lstm)
        utils.init_lstm_cell(self.buffer_lstm)
        utils.init_lstm_cell(self.action_lstm)
        utils.init_lstm_cell(self.stack_lstm)
        utils.init_lstm_cell(self.output_lstm)
        utils.init_lstm_cell(self.entity_forward_lstm)
        utils.init_lstm_cell(self.entity_backward_lstm)

    def forward(self, sentence, labels ,actions, hidden=None):

        sentence = sentence.squeeze(0)
        actions = actions.squeeze(0)
        labels = labels.squeeze(0)

        self.set_seq_size(sentence)
        word_embeds = self.dropout_e(self.word_embeds(sentence))
        action_embeds = self.dropout_e(self.action_embeds(actions))
        relation_embeds = self.dropout_e(self.relation_embeds(actions))

        action_count = 0

        buffer = StackRNN(self.buffer_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)
        stack = StackRNN(self.stack_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)
        action = StackRNN(self.action_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)
        output = StackRNN(self.output_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)
        ent_f = StackRNN(self.entity_forward_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)
        ent_b = StackRNN(self.entity_backward_lstm, self.gpu_triger, self.dropout, self._rnn_get_output, self.empty_emb)

        pre_actions = []
        losses = []
        right = 0
        sentence_array = sentence.data.cpu().numpy()
        token_embedding = list()

        for word_idx in range(len(sentence_array)):
            if sentence_array[word_idx] in self.singleton:
                if torch.rand(1).numpy()[0] < self.singleton_rate:
                    sentence_array[word_idx] = 0
            if self.use_spelling:
                if sentence_array[word_idx] == 0:

                    tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), self.unk_char_embeds], 1)
                else:
                    word = sentence_array[word_idx]
                    chars_in_word = [self.char2idx[char] for char in self.idx2word[word]]
                    chars_Tensor = utils.varible(torch.from_numpy(np.array(chars_in_word)), self.gpu_triger)
                    chars_embeds = self.dropout_e(self.char_embeds(chars_Tensor))
                    if self.char_structure == 'lstm':
                        char_o, hidden = self.char_bi_lstm(chars_embeds.unsqueeze(1), hidden)
                        char_out = torch.chunk(hidden[0].squeeze(1), 2, 0)
                        tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), char_out[0], char_out[1]], 1)
                    elif self.char_structure =='cnn':
                        char = chars_embeds.unsqueeze(0)
                        char = char.transpose(1, 2)
                        char, _ = self.conv1d(char).max(dim=2)
                        char = torch.tanh(char)
                        tok_rep = torch.cat([word_embeds[word_idx].unsqueeze(0), char], 1)
            else:
                tok_rep = word_embeds[word_idx].unsqueeze(0)
            if word_idx == 0:
                token_embedding = tok_rep
            else:
                token_embedding = torch.cat([token_embedding, tok_rep], 0)

        for i in range(len(token_embedding)):
            tok_embedding = token_embedding[len(token_embedding)-1-i].unsqueeze(0)
            tok = sentence.data[len(token_embedding)-1-i]
            buffer.push(tok_embedding, (tok_embedding, self.idx2word[tok]))

        while len(buffer) > 0 or len(stack) > 0:
            valid_actions = self.get_possible_actions(stack, buffer)

            # log_value
            log_probs = None
            if len(valid_actions)>1:

                lstms_output = torch.cat([buffer.embedding(), stack.embedding(), output.embedding(), action.embedding()], 1)
                hidden_output = torch.tanh(self.lstms_output_2_softmax(self.dropout(lstms_output)))
                if self.gpu_triger is True:
                    logits = self.output_2_act(hidden_output)[0][torch.autograd.Variable(torch.LongTensor(valid_actions)).cuda()]
                else:
                    logits = self.output_2_act(hidden_output)[0][torch.autograd.Variable(torch.LongTensor(valid_actions))]
                valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                log_probs = torch.nn.functional.log_softmax(logits)
                action_idx = torch.max(log_probs.cpu(), 0)[1][0].data.numpy()[0]
                action_predict = valid_actions[action_idx]
                pre_actions.append(action_predict)
                if log_probs is not None:
                    losses.append(log_probs[valid_action_tbl[actions.data[action_count]]])

            real_action = self.idx2action[actions.data[action_count]]
            if real_action == self.idx2action[action_predict]:
                right += 1
            act_embedding = action_embeds[action_count].unsqueeze(0)
            rel_embedding = relation_embeds[action_count].unsqueeze(0)
            action.push(act_embedding,(act_embedding, real_action))
            if real_action.startswith('S'):
                assert len(buffer) > 0
                tok_buffer_embedding, buffer_token = buffer.pop()
                stack.push(tok_buffer_embedding, (tok_buffer_embedding, buffer_token))
            elif real_action.startswith('O'):
                assert len(buffer) > 0
                tok_buffer_embedding, buffer_token = buffer.pop()
                output_input = self.buffer_2_output(torch.cat([tok_buffer_embedding, rel_embedding],1))
                output.push(output_input, (tok_buffer_embedding, buffer_token))
            elif real_action.startswith('R'):
                ent =''
                entity = []
                assert len(stack) > 0
                while len(stack) > 0:
                    tok_stack_embedding, stack_token = stack.pop()
                    entity.append([tok_stack_embedding, stack_token])
                if len(entity) > 1:

                    for i in range(len(entity)):
                        ent_f.push(entity[i][0], (entity[i][0],entity[i][1]))
                        ent_b.push(entity[len(entity)-i-1][0], (entity[len(entity)-i-1][0], entity[len(entity)-i-1][1]))
                        ent += entity[i][1]
                        ent += ' '
                    entity_input = self.dropout(torch.cat([ent_f.embedding(), ent_b.embedding()], 1))
                else:
                    ent_f.push(entity[0][0],(entity[0][0], entity[0][1]))
                    ent_b.push(entity[0][0],(entity[0][0], entity[0][1]))
                    ent = entity[0][1]
                    entity_input = self.dropout(torch.cat([ent_f.embedding(), ent_b.embedding()], 1))
                ent_b.clear()
                ent_f.clear()
                output_input = self.entity_2_output(torch.cat([entity_input, rel_embedding], 1))
                output.push(output_input, (entity_input, ent))
            action_count += 1

        loss = -torch.sum(torch.cat(losses))

        return loss, pre_actions, right if len(losses) > 0 else None