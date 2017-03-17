import codecs
import numpy as np
from os import path
import gc
import re
from collections import Counter, OrderedDict, namedtuple

encoding = 'utf8'
# encoding='iso-8859-1'

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])

def vocab_unpack(vocab):
    return vocab['idx2word'], vocab['word2idx'], vocab['idx2char'], vocab['char2idx']


def load_data(tokens, data_dir, max_word_l, n_words, n_chars):
    n_words = n_words
    n_chars = n_chars

    train_file = path.join(data_dir, 'train.txt')
    valid_file = path.join(data_dir, 'valid.txt')
    test_file = path.join(data_dir, 'test.txt')
    input_files = [train_file, valid_file, test_file]
    vocab_file = path.join(data_dir, 'vocab.npz')
    tensor_file = path.join(data_dir, 'data')
    char_file = path.join(data_dir, 'data_char')

    # construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)):
        print 'one-time setup: preprocessing input train/valid/test files in dir: ', data_dir
        text_to_tensor(tokens, input_files, vocab_file, tensor_file, char_file, max_word_l, n_words, n_chars)

    print('loading data files...')
    all_data = []
    all_data_char = []
    for split in range(3):
        all_data.append(np.load("{}_{}.npy".format(tensor_file, split)))  # train, valid, test tensors
        all_data_char.append(np.load("{}_{}.npy".format(char_file, split)))  # train, valid, test character indices
    vocab_mapping = np.load(vocab_file)
    idx2word, word2idx, idx2char, char2idx = vocab_unpack(vocab_mapping)
    word_vocab_size = len(idx2word)
    char_vocab_size = len(idx2char)
    print 'Word vocab size: %d, Char vocab size: %d' % (len(idx2word), len(idx2char))
    # create word-char mappings
    max_word_l = all_data_char[0].shape[1]
    # cut off the end for train/valid sets so that it divides evenly
    # test set is not cut off
    return all_data, all_data_char, max_word_l, word_vocab_size, char_vocab_size

    seq_length = seq_length
    data_sizes = []
    split_sizes = []
    all_batches = []
    print('reshaping tensors...')
    for split, data in enumerate(all_data):
        print 'data shape: ', data.shape
        data_len = data.shape[0]
        data_sizes.append(data_len)
        if split < 2 and data_len % (batch_size * seq_length) != 0:
            data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
        ydata = data.copy()
        ydata[0:-1] = data[1:]
        ydata[-1] = data[0]
        data_char = all_data_char[split][:len(data)]
        if split < 2:
            rdata = data.reshape((batch_size, -1))
            rydata = ydata.reshape((batch_size, -1))
            rdata_char = data_char.reshape((batch_size, -1, max_word_l))
        else:  # for test we repeat dimensions to batch size (easier but inefficient evaluation)
            nseq = (data_len + (seq_length - 1)) // seq_length
            rdata = data.copy()
            rdata.resize((1, nseq * seq_length))
            rdata = np.tile(rdata, (batch_size, 1))
            rydata = ydata.copy()
            rydata.resize((1, nseq * seq_length))
            rydata = np.tile(rydata, (batch_size, 1))
            rdata_char = data_char.copy()
            rdata_char.resize((1, nseq * seq_length, rdata_char.shape[1]))
            rdata_char = np.tile(rdata_char, (batch_size, 1, 1))
        # split in batches
        x_batches = np.split(rdata, rdata.shape[1] / seq_length, axis=1)
        y_batches = np.split(rydata, rydata.shape[1] / seq_length, axis=1)
        x_char_batches = np.split(rdata_char, rdata_char.shape[1] / seq_length, axis=1)
        print 'x_batches: ', x_char_batches[0].shape
        nbatches = len(x_batches)
        split_sizes.append(nbatches)
        assert len(x_batches) == len(y_batches)
        assert len(x_batches) == len(x_char_batches)
        all_batches.append((x_batches, y_batches, x_char_batches))

    batch_idx = [0, 0, 0]
    word_vocab_size = len(idx2word)
    print 'data load done. Number of batches in train: %d, val: %d, test: %d' \
          % (split_sizes[0], split_sizes[1], split_sizes[2])


def text_to_tensor(tokens, input_files, out_vocabfile, out_tensorfile, out_charfile, max_word_l, n_words, n_chars):
    print 'Processing text into tensors...'
    max_word_l_tmp = 0  # max word length of the corpus
    idx2word = [tokens.UNK]  # unknown word token
    word2idx = OrderedDict()
    word2idx[tokens.UNK] = 0
    idx2char = [tokens.ZEROPAD, tokens.START, tokens.END, tokens.UNK]  # zero-pad, start-of-word, end-of-word tokens
    char2idx = OrderedDict()
    char2idx[tokens.ZEROPAD] = 0
    char2idx[tokens.START] = 1
    char2idx[tokens.END] = 2
    char2idx[tokens.UNK] = 3
    split_counts = []

    # first go through train/valid/test to get max word length
    # if actual max word length is smaller than specified
    # we use that instead. this is inefficient, but only a one-off thing so should be fine
    # also counts the number of tokens
    prog = re.compile('\s+')
    wordcount = Counter()
    charcount = Counter()
    for split in range(3):  # split = 0 (train), 1 (val), or 2 (test)

        def update(word):
            if word[0] == tokens.UNK:
                if len(word) > 1:  # unk token with character info available
                    word = word[2:]
            else:
                wordcount.update([word])
            word = word.replace(tokens.UNK, '')
            charcount.update(word)

        f = codecs.open(input_files[split], 'r', encoding)
        counts = 0
        for line in f:
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            words = prog.split(line)
            for word in filter(None, words):
                update(word)
                max_word_l_tmp = max(max_word_l_tmp, len(word) + 2)  # add 2 for start/end chars
                counts += 1
            if tokens.EOS != '':
                update(tokens.EOS)
                counts += 1  # PTB uses \n for <eos>, so need to add one more token at the end
        f.close()
        split_counts.append(counts)

    print 'Most frequent words:', len(wordcount)
    for ii, ww in enumerate(wordcount.most_common(n_words - 1)):
        word = ww[0]
        word2idx[word] = ii + 1
        idx2word.append(word)
        if ii < 3: print word

    print 'Most frequent chars:', len(charcount)
    for ii, cc in enumerate(charcount.most_common(n_chars - 4)):
        char = cc[0]
        char2idx[char] = ii + 4
        idx2char.append(char)
        if ii < 3: print char

    print 'Char counts:'
    for ii, cc in enumerate(charcount.most_common()):
        print ii, cc[0].encode(encoding), cc[1]

    print 'After first pass of data, max word length is: ', max_word_l_tmp
    print 'Token count: train %d, val %d, test %d' % (split_counts[0], split_counts[1], split_counts[2])


    max_word_l = max(max_word_l_tmp, max_word_l)

    for split in range(3):  # split = 0 (train), 1 (val), or 2 (test)
        # Preallocate the tensors we will need.
        # Watch out the second one needs a lot of RAM.
        output_tensor = np.empty(split_counts[split], dtype='int32')
        output_chars = np.zeros((split_counts[split], max_word_l), dtype='int32')

        def append(word, word_num):
            chars = [char2idx[tokens.START]]  # start-of-word symbol
            if word[0] == tokens.UNK and len(word) > 1:  # unk token with character info available
                word = word[2:]
                output_tensor[word_num] = word2idx[tokens.UNK]
            else:
                output_tensor[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
            chars += [char2idx[char] for char in word if char in char2idx]
            chars.append(char2idx[tokens.END])  # end-of-word symbol
            if len(chars) >= max_word_l:
                chars[max_word_l - 1] = char2idx[tokens.END]
                output_chars[word_num] = chars[:max_word_l]
            else:
                output_chars[word_num, :len(chars)] = chars
            return word_num + 1

        f = codecs.open(input_files[split], 'r', encoding)
        word_num = 0
        for line in f:
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            words = prog.split(line)
            for rword in filter(None, words):
                word_num = append(rword, word_num)
            if tokens.EOS != '':  # PTB does not have <eos> so we add a character for <eos> tokens
                word_num = append(tokens.EOS, word_num)  # other datasets don't need this
        f.close()
        tensorfile_split = "{}_{}.npy".format(out_tensorfile, split)
        print 'saving ', tensorfile_split
        np.save(tensorfile_split, output_tensor)
        charfile_split = "{}_{}.npy".format(out_charfile, split)
        print 'saving ', charfile_split
        np.save(charfile_split, output_chars)

    # save output preprocessed files
    print 'saving ', out_vocabfile
    np.savez(out_vocabfile, idx2word=idx2word, word2idx=word2idx, idx2char=idx2char, char2idx=char2idx)

if __name__ == '__main__':
    tokens = Tokens(
        EOS='+',
        UNK='|',  # unk word token
        START='{',  # start-of-word token
        END='}',  # end-of-word token
        ZEROPAD=' '  # zero-pad token
    )
    load_data(tokens, data_dir='data/ptb', batch_size=100, seq_length=100, max_word_l=10, n_words=100000, n_chars=100)
