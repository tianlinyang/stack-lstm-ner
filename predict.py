from __future__ import print_function
import codecs
# from model.stack_lstm import *
from model.batch_stack_lstm import *
import model.utils as utils
import model.evaluate as evaluate

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating Stack-LSTM')
    parser.add_argument('--load_arg', default='./checkpoint/ner_2018-06-22/dev=68.33.json',
                        help='arg json file path')
    parser.add_argument('--spelling', default=True, help='use spelling or not')
    parser.add_argument('--load_check_point', default='./checkpoint/ner_2018-06-22/dev=68.33.model',
                        help='checkpoint path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--mode', choices=['train', 'predict'], default='predict', help='mode selection')
    parser.add_argument('--test_file', default='test.txt',
                        help='path to test file, if set to none, would use test_file path in the checkpoint file')
    parser.add_argument('--test_file_out', default='test_out.txt',
                        help='path to test file output, if set to none, would use test_file path in the checkpoint file')
    args = parser.parse_args()

    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    a_map = checkpoint_file['a_map']
    char_map = checkpoint_file['char_map']
    singleton = checkpoint_file['singleton']
    ner_map = checkpoint_file['ner_map']
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # load corpus
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()


    # converting format
    test_features = utils.read_corpus_predict(test_lines)

    # construct dataset
    test_dataset = utils.construct_dataset_predict(test_features, f_map, jd['caseless'])

    test_dataset_loader = [torch.utils.data.DataLoader(test_dataset, args.batch_size,  shuffle=False, drop_last=False)]

    # build model
    ner_model = TransitionNER(args.mode, a_map, f_map, l_map, char_map, ner_map, len(f_map), len(a_map), jd['embedding_dim'], jd['action_embedding_dim'], jd['char_embedding_dim'], jd['hidden'], jd['char_hidden'],
                              jd['layers'], jd['drop_out'], args.spelling, jd['char_structure'], is_cuda=args.gpu)
    print("loading model")
    ner_model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
    else:
        if_cuda = False
    file_out = codecs.open(args.test_file_out, "w+", encoding="utf-8")
    print("decoding")
    evaluate.generate_ner(ner_model, file_out, test_dataset_loader, a_map, f_map, if_cuda)


