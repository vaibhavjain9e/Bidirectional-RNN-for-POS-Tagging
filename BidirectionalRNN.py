import numpy
import tensorflow as tf
import sys

TRAIN_TIME_MINUTES = 8


class DatasetReader(object):

    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.

        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!

        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...]
        """
        training_data = open(filename, mode="r", encoding="utf-8").read().split("\n")
        term_list = []
        tag_list = []

        training_data = list(filter(None, training_data))

        for item in training_data:
            item2 = item.split(" ")
            for element in item2:
                word_tag = element.rsplit('/', 1)
                term_list.append(word_tag[0])
                tag_list.append(word_tag[1])

        term_list = numpy.array(term_list)
        _, idx1 = numpy.unique(term_list, return_index=True)
        term_list = term_list[numpy.sort(idx1)]

        tag_list = numpy.array(tag_list)
        _, idx2 = numpy.unique(tag_list, return_index=True)
        tag_list = tag_list[numpy.sort(idx2)]

        if len(term_index) > 0:
            max_term_value = max(term_index.values()) + 1
        else:
            max_term_value = 0
        if len(tag_index) > 0:
            max_tag_value = max(tag_index.values()) + 1
        else:
            max_tag_value = 0

        for item in term_list:
            if item not in term_index:
                term_index[item] = max_term_value
                max_term_value = max_term_value + 1

        for item in tag_list:
            if item not in tag_index:
                tag_index[item] = max_tag_value
                max_tag_value = max_tag_value + 1

        parsed_file = []
        for item in training_data:
            item2 = item.split(" ")
            parsed_line = []
            for element in item2:
                word_tag = element.rsplit('/', 1)
                parsed_line.append((term_index[word_tag[0]], tag_index[word_tag[1]]))
            parsed_file.append(parsed_line)

        return parsed_file


    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]],

                [2, 4]
            )
        """
        N = len(dataset)
        T = len(max(dataset, key=len))
        terms_matrix = numpy.zeros((N, T))
        tags_matrix = numpy.zeros((N, T))
        lengths = numpy.zeros(N)

        for i in range(N):
            for j in range(len(dataset[i])):
                terms_matrix[i][j] = dataset[i][j][0]
                tags_matrix[i][j] = dataset[i][j][1]
            lengths[i] = len(dataset[i])

        terms_matrix = terms_matrix.astype(int)
        tags_matrix = tags_matrix.astype(int)
        lengths = lengths.astype(int)
        tuple_values = (terms_matrix, tags_matrix, lengths)

        return tuple_values

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                    (train_terms, train_tags, train_lengths),
                    (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """
        The arguments are arbitrary: pass them from main().

        Args:
            max_length: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).
        """
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.y = tf.placeholder(tf.int32, [None, self.max_length], name='y')
        self.lengths = tf.placeholder(tf.int64, [None], 'lengths')
        self.lr = 0.017
        self.session = tf.Session()

    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        length_vector = [2, 5, 3]
        return mask = [[1,1,0,0,0], [1,1,1,1,1], [1,1,1,0,0]]
        """
        return tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)

    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths .made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """

        logits = self.session.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

    def build_training(self):
        """Prepares the class for training."""

        def vanilla_rnn():
            states = []
            state_size = 32
            embedding_size = 32

            # Create Embedding
            embeddings = tf.get_variable('embeddings', [self.num_terms, embedding_size])
            xemb = tf.nn.embedding_lookup(embeddings, self.x)

            # RNN Layer
            rnn_cell = tf.keras.layers.SimpleRNNCell(state_size)
            cur_state = tf.zeros(shape=[1, state_size])
            for i in range(self.max_length):
                cur_state = rnn_cell(xemb[:, i, :], [cur_state])[0]
                states.append(cur_state)

            stacked_states = tf.stack(states, axis=1)

            def fully_connected_layer():
                l2_reg = tf.contrib.layers.l2_regularizer(1e-6)
                self.logits = tf.contrib.layers.fully_connected(
                    stacked_states, self.num_tags, activation_fn=tf.nn.relu, weights_regularizer=l2_reg)

            def dense_layer():
                self.logits = tf.layers.dense(stacked_states, self.num_tags)
                self.logits = tf.reshape(self.logits, [-1, self.max_length, self.num_tags])

            dense_layer()
            # fully_connected_layer()

        def bidirectional_rnn():
            state_size = 104
            embedding_size = 52

            # embeddings = [batch_size X embedding_size]
            embeddings = tf.get_variable('embeddings', [self.num_terms, embedding_size])

            # RNN Input
            # inputs =  [batch_size X max_length X embedding_size]
            inputs = tf.nn.embedding_lookup(embeddings, self.x)

            """
            Bidirectional LSTM Layer
            It reads the input from left to right and right to left and gives
            2 outputs (output forward and output backward) which are concatenated
            together to form the final output.
            tf.contrib.rnn.stack_bidirectional_rnn can be used instead of tf.nn.bidirectional_dynamic_rnn
            The later one is costlier but more efficient
            """

            # Create LSTM cell
            rnn_cell = tf.nn.rnn_cell.LSTMCell(state_size)

            # output_fw and output_bw = [batch_size X max_length X state_size]
            (output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell,
                cell_bw=rnn_cell,
                inputs=inputs,
                sequence_length=self.lengths,
                dtype=tf.float32)
            inputs = tf.concat([output_fw, output_bw], axis=-1)

            # Dense Layer with Linear activation Function(default)
            logit_inputs = tf.reshape(inputs, [-1, 2 * state_size])
            self.logits = tf.layers.dense(logit_inputs, self.num_tags)
            self.logits = tf.reshape(self.logits, [-1, self.max_length, self.num_tags])

        bidirectional_rnn()
        # vanilla_rnn()

        # Calculating Loss
        loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.y, self.lengths_vector_to_binary_matrix(self.lengths))
        tf.losses.add_loss(loss, loss_collection=tf.GraphKeys.LOSSES)

        self.learning_rate = tf.placeholder_with_default(
            numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        def no_clipping():
            self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), optimizer)

        def gradient_clipping():
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

        gradient_clipping()
        # no_clipping()
        self.session.run(tf.global_variables_initializer())

    def train_epoch(self, terms, tags, lengths, batch_size=32):
        """Performs updates on the model given training training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size.

        Return:
            boolean. Return True iff you want the training to continue. If
            you return False (or do not return anything) then training will stop after
            the first iteration!
        """
        learn_rate = self.lr
        self.lr = self.lr / 2

        def batch_step(batch_x, batch_y, batch_lengths, learn_rate):
            self.session.run(self.train_op, {
                self.x: batch_x,
                self.y: batch_y,
                self.learning_rate: learn_rate,
                self.lengths: batch_lengths
            })

        indices = numpy.random.permutation(terms.shape[0])
        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]] + 0  # + 0 to copy slice
            batch_step(slice_x, tags[indices[si:se]], lengths[indices[si:se]], learn_rate)

        return True

    def evaluate(self, terms, tags, lengths):
        predicted_tags = self.run_inference(terms, lengths)

        if predicted_tags is None:
            print('Is your run_inference function implented?')
            return 0

        test_accuracy = numpy.sum(numpy.cumsum(numpy.equal(tags, predicted_tags), axis=1)[
                                      numpy.arange(lengths.shape[0]), lengths - 1]) / numpy.sum(lengths + 0.0)
        print(test_accuracy)



def main():
    # Read dataset.
    reader = DatasetReader
    # train_filename = "D:\\hmm-training-data\\ja_gsd_train_tagged.txt"
    # #train_filename = "D:\\hmm-training-data\\it_isdt_train_tagged.txt"
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j + 1))
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()
