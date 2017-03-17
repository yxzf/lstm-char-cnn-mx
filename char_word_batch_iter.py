#!encoding=utf-8
import numpy as np
import mxnet as mx


class CharWordBatch(object):
    def __init__(self, data_char, data_word, label, init_state_arrays):
        self.data = [mx.nd.array(data_char)] + init_state_arrays
        self.label = [mx.nd.array(label)]

class CharWordBatchIter(mx.io.DataIter):
    def __init__(self, data, data_char, batch_size, seq_length, max_word_length, init_states, data_name='data', label_name='label'):
        super(CharWordBatchIter, self).__init__()

        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size

        data_len = data.shape[0]
        data = data[:batch_size * seq_length * (data_len // (batch_size * seq_length))]
        ydata = data.copy()
        ydata[0:-1] = data[1:]
        ydata[-1] = data[0]
        data_char = data_char[:len(data)]

        rdata = data.reshape((batch_size, -1))
        rydata = ydata.reshape((batch_size, -1))
        rdata_char = data_char.reshape((batch_size, -1, max_word_length))

        # split in batches
        self.word_batches = np.split(rdata, rdata.shape[1] / seq_length, axis=1)
        self.label_batches = np.split(rydata, rydata.shape[1] / seq_length, axis=1)
        self.char_batches = np.split(rdata_char, rdata_char.shape[1] / seq_length, axis=1)

        self.num_batch = len(self.word_batches)
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        # self.provide_data = [('data', (batch_size, seq_length, max_word_length))] #('word_data', (batch_size, seq_length)),
        # self.provide_label = [('label', (batch_size, seq_length))]
        self.provide_data = [('data', (batch_size, seq_length, max_word_length))] + init_states
        self.provide_label = [('label', (batch_size, seq_length))]

        self.cursor_index = -1

    def get_word(self):
        return self.word_batches[self.cursor_index]

    def get_char(self):
        return self.char_batches[self.cursor_index]

    def get_label(self):
        return self.label_batches[self.cursor_index]


    def iter_next(self):
        self.cursor_index += 1
        return self.cursor_index < self.num_batch

    # 此处实现迭代
    def next(self):
        if self.iter_next():
            return CharWordBatch(self.get_char(), self.get_word(), self.get_label(), self.init_state_arrays)

        else:
            raise StopIteration

    def reset(self):
        self.cursor_index = 0
