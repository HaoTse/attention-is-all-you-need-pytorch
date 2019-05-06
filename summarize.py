''' Summarize input text with trained model. '''

import torch
import torch.utils.data
import argparse

import transformer.Constants as Constants

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import convert_instance_to_idx_seq


def read_instances(input_sent, max_sent_len, keep_case, mode=None):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent = False

    if not keep_case:
        input_sent = input_sent.lower()

    if mode == 'char':
        input_sent = [w for w in input_sent if w.strip()]
        input_sent = ' '.join(input_sent)

    words = input_sent.split()
    if len(words) > max_sent_len:
        trimmed_sent = True
    word_inst = words[:max_sent_len]

    if word_inst:
        word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
    else:
        word_insts += [None]

    if trimmed_sent > 0:
        print('[Warning] instances is trimmed to the max sentence length {}.'
              .format(max_sent_len))

    return word_insts


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='summarize.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    # parser.add_argument('-src', required=True,
    #                     help='Input sentence')
    parser.add_argument('-vocab', required=True,
                        help='Path to vocabulary .pt file')
    parser.add_argument('-output', default='',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('-server_port', type=str, default='', help="Can be used for distant debugging.")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    opt.src = '本文 總結 了 十個 可 穿戴 產品 的 設計 原則'

    if opt.server_ip and opt.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(opt.server_ip, opt.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case,
        preprocess_settings.mode)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=1,
        collate_fn=collate_fn)

    translator = Translator(opt)

    # run model
    for batch in test_loader:
        all_hyp, all_scores = translator.translate_batch(*batch)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])

    if opt.output:
        with open(opt.output, 'w') as f:
            f.write(pred_line + '\n')
    else:
        print(pred_line)

    print('[Info] Finished.')

if __name__ == "__main__":
    main()
