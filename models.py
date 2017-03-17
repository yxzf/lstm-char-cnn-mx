#!/usr/bin/env python
#encoding=utf-8
import mxnet as mx
import numpy as np
import lstm

def cnn_for_sentence_classification(batch_size, vocab_size, embed_size, num_label, with_embedding,
                                    sentence_size, dropout=0.5, filter_list = [2, 3, 4], num_filter=100):
    input_x = mx.sym.Variable(name='data')
    input_y = mx.sym.Variable(name='softmax_label')
    # mxnet 中conv的输入格式是四维(batch_size, channel_size, height, width) ()

    if not with_embedding:
        embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=embed_size)
        conv_input = mx.sym.Reshape(data=embed_layer, shape=[batch_size, 1, sentence_size, embed_size])
    else:
        conv_input = input_x

    conv_list = []
    for i, filter_size in enumerate(filter_list):
        conv = mx.sym.Convolution(conv_input, kernel=(filter_size, embed_size),
                                   num_filter=num_filter)
        relu = mx.sym.Activation(conv, act_type='relu')
        pool = mx.sym.Pooling(relu, pool_type='max', kernel=(sentence_size-filter_size+1, 1), stride=(1, 1))
        conv_list.append(pool)
    filter_total = num_filter * len(filter_list)
    concat = mx.sym.Concat(*conv_list, dim=1)
    concat_reshape = mx.sym.Reshape(concat, shape=(batch_size, filter_total))

    fc1 = mx.sym.FullyConnected(concat_reshape, num_hidden=100)
    relu1 = mx.sym.Activation(fc1, act_type='relu')
    dropout1 = mx.sym.Dropout(relu1, p=dropout)
    fc2 = mx.sym.FullyConnected(dropout1, num_hidden=num_label)
    sm = mx.sym.SoftmaxOutput(data=fc2, label=input_y)
    return sm

def char_cnn_for_sentence_classification(char_size, filter_list, num_filter, num_class=2, dropout=0.5):
    input_x = mx.sym.Variable(name='data') # (batch_size, 1, 70, 1014)
    input_y = mx.sym.Variable(name='softmax_label')



    layer_num = 0
    conv_1 = mx.sym.Convolution(input_x, kernel=(char_size, filter_list[layer_num]), num_filter=num_filter)
    act_1 = mx.sym.Activation(conv_1, act_type='relu')
    pool_1 = mx.sym.Pooling(act_1, pool_type='max', kernel=(1, 3), stride=(1, 3))
    layer_num += 1

    conv_2 = mx.sym.Convolution(pool_1, kernel=(1, filter_list[layer_num]), num_filter=num_filter)
    act_2 = mx.sym.Activation(conv_2, act_type='relu')
    pool_2 = mx.sym.Pooling(act_2, pool_type='max', kernel=(1, 3), stride=(1, 3))
    layer_num += 1

    conv_3 = mx.sym.Convolution(pool_2, kernel=(1, filter_list[layer_num]), num_filter=num_filter)
    act_3 = mx.sym.Activation(conv_3, act_type='relu')
    layer_num += 1

    conv_4 = mx.sym.Convolution(act_3, kernel=(1, filter_list[layer_num]), num_filter=num_filter)
    act_4 = mx.sym.Activation(conv_4, act_type='relu')
    layer_num += 1

    conv_5 = mx.sym.Convolution(act_4, kernel=(1, filter_list[layer_num]), num_filter=num_filter)
    act_5 = mx.sym.Activation(conv_5, act_type='relu')
    layer_num += 1

    conv_6 = mx.sym.Convolution(act_5, kernel=(1, filter_list[layer_num]), num_filter=num_filter)
    act_6 = mx.sym.Activation(conv_6, act_type='relu')
    pool_6 = mx.sym.Pooling(act_6, pool_type='max', kernel=(1, 3), stride=(1, 3))

    flatten = mx.sym.Flatten(pool_6)

    fc1 = mx.sym.FullyConnected(flatten, num_hidden=1024)
    relu1 = mx.sym.Activation(fc1, act_type='relu')
    dropout1 = mx.sym.Dropout(relu1, p=dropout)

    fc2 = mx.sym.FullyConnected(dropout1, num_hidden=1024)
    relu2 = mx.sym.Activation(fc2, act_type='relu')
    dropout2 = mx.sym.Dropout(relu2, p=dropout)
    fc3 = mx.sym.FullyConnected(dropout2, num_hidden=num_class)
    sm = mx.sym.SoftmaxOutput(data=fc3, label=input_y)
    return sm


def highway(input, highway_layer, highway_g, name_scope):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y

    where g is nonlinearity(like tanh、relu), t is transform gate, and (1 - t) is carry gate.
    """
    output = input
    for layer_num in range(highway_layer):
        # print 'seq: ', name_scope, 'layer_num: ', layer_num
        t_weight = mx.sym.Variable("%d_layer_%d_t_weight" %(name_scope, layer_num))
        t_bias = mx.sym.Variable("%d_layer_%d_t_bias" % (name_scope, layer_num))


        l_weight = mx.sym.Variable('%d_layer_%d_l_weight' %(name_scope, layer_num))
        l_bias = mx.sym.Variable('%d_layer_%d_l_bias' %(name_scope, layer_num))

        trainsform_linear = output * t_weight + t_bias
        transform_gate = mx.sym.Activation(trainsform_linear, act_type='sigmoid')
        nonlinear_linear = output * l_weight + l_bias
        nonlinear_input = mx.sym.Activation(nonlinear_linear, act_type=highway_g)
        output = transform_gate * nonlinear_input + (1 - transform_gate) * input
    return output

def word_char_cnn_rnn(batch_size, seq_length, word_vocab_size, char_vocab_size, char_embed_size,
                      max_word_length, filter_sizes, feature_map, num_lstm_layer,
                      num_hidden, highway_layer=2, highway_g='relu', use_batch_norm=True, dropout=0.5):


    input_char = mx.sym.Variable(name='data')   # shape (batch_size, seq_length, max_word_length)
    label = mx.sym.Variable(name='label')       # shape (batch_size, seq_length)

    # embed_layer shape (batch_size, seq_length, max_word_length, char_embed_size)
    embed_layer = mx.sym.Embedding(data=input_char, input_dim=char_vocab_size, output_dim=char_embed_size)
    # split embed_layer into number of seq_length, each shape is (batch_size, max_word_length, char_embed_size)
    seq_symbols = mx.sym.SliceChannel(data=embed_layer, num_outputs=seq_length, axis=1)

    rnn_inputs = []
    for i in range(seq_length):
        #print 'seq_length: ', i
        # for each word with char matrix
        conv_input = mx.sym.Reshape(seq_symbols[i], shape=(batch_size, 1, max_word_length, char_embed_size))
        conv_pools = []
        for filter_size, num_filter in zip(filter_sizes, feature_map):
            conv = mx.sym.Convolution(conv_input, kernel=(filter_size, char_embed_size), num_filter=num_filter)
            relu = mx.sym.Activation(conv, act_type='relu')
            pool = mx.sym.Pooling(relu, pool_type='max', kernel=(max_word_length-filter_size+1, 1), stride=(1, 1))
            conv_pools.append(pool)
        filter_total = num_filter * len(filter_sizes)
        concat = mx.sym.Concat(*conv_pools, dim=1)
        cnn_output = mx.sym.Reshape(data=concat, shape=(batch_size, filter_total))
        if use_batch_norm:
            cnn_output = mx.sym.BatchNorm(data=cnn_output, fix_gamma=False, eps=2e-5, momentum=0.9)

        if highway_layer == 0:
            rnn_input = cnn_output
        else:
            rnn_input = highway(cnn_output, highway_layer, highway_g, i)
        rnn_inputs.append(rnn_input)

    return lstm.lstm_unroll(rnn_inputs, label, num_lstm_layer=num_lstm_layer, seq_len=seq_length,
                     input_size=num_filter * len(filter_sizes), num_hidden=num_hidden, num_embed=10,
                            num_label=word_vocab_size, dropout=dropout)














