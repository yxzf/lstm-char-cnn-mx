#!/usr/bin/env python
import mxnet as mx
import numpy as np
from char_word_data_helper import *
from char_word_batch_iter import *
from models import *


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    #print np.exp(loss / label.size)
    return np.exp(loss / label.size)


def fun():
    tokens = Tokens(
        EOS='+',
        UNK='|',  # unk word token
        START='{',  # start-of-word token
        END='}',  # end-of-word token
        ZEROPAD=' '  # zero-pad token
    )
    batch_size = 128
    seq_length = 35

    char_embed_size = 15

    filter_sizes = [1, 2, 3, 4, 5]
    feature_map = [25, 25, 25, 25, 25]

    num_hidden = 300
    num_lstm_layer = 2
    highway_layer = 2
    highway_g = 'relu'


    all_data, all_data_char, max_word_length, word_vocab_size, char_vocab_size = \
        load_data(tokens, data_dir='data/ptb', max_word_l=10, n_words=100000, n_chars=100)

    print 'word_vocab_size', word_vocab_size
    print 'char_vocab_size', char_vocab_size
    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h


    train_iter = CharWordBatchIter(data=all_data[0], data_char=all_data_char[0], batch_size=batch_size,
                      seq_length=seq_length, max_word_length=max_word_length, init_states=init_states)

    val_iter = CharWordBatchIter(data=all_data[1], data_char=all_data_char[1], batch_size=batch_size,
                      seq_length=seq_length, max_word_length=max_word_length, init_states=init_states)


    model = word_char_cnn_rnn(batch_size=batch_size, seq_length=seq_length, word_vocab_size = word_vocab_size,
                              char_vocab_size=char_vocab_size, char_embed_size=char_embed_size,
                              max_word_length=max_word_length, filter_sizes=filter_sizes, feature_map=feature_map,
                              num_lstm_layer=num_lstm_layer, num_hidden=num_hidden, highway_layer=highway_layer,
                              highway_g=highway_g)
    #model = mx.mod.Module(model)
    contexts = [mx.context.cpu(i) for i in range(1)]


    # mx.metric.EvalMetric
    # eval_metric.append()

    print 'model start===>'

    # model.fit(train_iter, initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
    #           eval_data=val_iter, optimizer='rmsprop', optimizer_params={'learning_rate': 0.0001},
    #           num_epoch=10)


    lr_scheduler = mx.lr_scheduler.FactorScheduler(10000, 0.5)
    clip_gradient = 0.1
    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=model,
                                 num_epoch=100,
                                 learning_rate=0.01,
                                 momentum=0.9,
                                 wd=0.00001)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=train_iter, eval_data=val_iter,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 20), )


if __name__ == '__main__':
    fun()








