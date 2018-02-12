import tensorflow as tf 
import numpy as np
import helpers
import operator
import random
import argparse
import data_functions
import parameters_class


# create batches with size of batch_size
def create_batches(source_data, target_data, parameters):
    # stores batches
    source_batches = []
    target_batches = []
    # stores last batch ending index
    prev_batch_end = 0
    
    for j in range(0, len(source_data)):
	# if it's a full batch
        if j % parameters.batch_size == 0 and j != 0:
            # stores a batch
            sbatch = []
            tbatch = []
            for k in range(prev_batch_end+1,j+1):
                # store sequence
                sbatch.append(source_data[k][0])
                # store expected target_data (known from source_data index)
                tbatch.append(target_data[source_data[k][1]])
            # add created batch
            source_batches.append(sbatch)
            target_batches.append(tbatch)
            prev_batch_end = j
            
    # put the rest of it in another batch
    if prev_batch_end != j:
        sbatch = []
        tbatch = []
        for k in range(prev_batch_end+1,j):
            sbatch.append(source_data[k][0])
            tbatch.append(target_data[source_data[k][1]])
        source_batches.append(sbatch)
        target_batches.append(tbatch)

    # in case its a single line
    if j == 0: 
        source_batches.append([source_data[j][0]])
        target_batches.append([target_data[source_data[j][1]]])

    return source_batches, target_batches


# feed encoder with the sequences of the next batch
def next_feed(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, batch_num, parameters, learning_rate):
        # get transpose of source_batches[batch_num]
        encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_batches[batch_num])
    
        # get max input sequence length
        max_input_length = max(encoder_input_lengths_)
    
        # target word is max character_changing_num character longer than source word 
        # get transpose of target_batches[i] and put an EOF and PAD at the end
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in target_batches[batch_num]]
        )
        
        '''
        # learning rate decay
        if batch_num < 50:
            parameters.learning_rate = 1.0
        if batch_num < 100:
            parameters.learning_rate = 0.1
        if batch_num < 150:
            parameters.learning_rate = 0.01
        if batch_num >= 150:
            parameters.learning_rate = 0.001
        '''

        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_ #,
            #learning_rate: parameters.learning_rate
        }


# train the model with chosen parameters
def train_model(source_data, target_data, encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, sess, loss_track, parameters, saver, alphabet_and_morph_tags, exp_num, learning_rate, summ, writer):
    # early stopping patience
    patience_counter = 0

    for epoch_num in range(parameters.epoch):
            print('Epoch:',epoch_num)
            epoch_loss = 0
            
            # shuffle it in every epoch for creating random batches
            source_data = random.sample(source_data, len(source_data))
        	
            # encoder inputs and decoder outputs devided into batches
            source_batches, target_batches = create_batches(source_data, target_data, parameters)
            
            # get every batches and train the model on it
            for batch_num in range(0, len(source_batches)):
                fd = next_feed(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, batch_num, parameters, learning_rate)
   
                _, l, s = sess.run([train_op, loss, summ], fd)
                epoch_loss += l

                # write summary to Tensorboard every 5 epoch
                if epoch_num % 5 == 0:
                    writer.add_summary(s, epoch_num)
                
            print('epoch:',epoch_num, 'loss:', epoch_loss)

            # store current epoch loss to calculate early stopping delta
            loss_track.append(epoch_loss)

            # early stopping
            if epoch_num > 0 and loss_track[epoch_num - 1] - loss_track[epoch_num] > parameters.early_stopping_delta:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > parameters.early_stopping_patience:
                break

    # log parametercombination's loss
    with open('parameters/' + 'trained_model' + str(exp_num) + '_parameters.tsv', 'a') as output_parameters:
        output_parameters.write('loss\t{}\n'.format(epoch_loss))

    # create model's directory if it's not existing
    directory = 'trained_models/trained_model' + str(exp_num)
    data_functions.create_directory(directory)

    # save the model
    saver.save(sess, 'trained_models/trained_model' + str(exp_num) + '/' + 'trained_model' + str(exp_num))
    return


# main function
def main():
    parameters = parameters_class.Parameters(2, 1, 0, 10, 300, 100, 100, 0.001, 5, 20, 0.001)

    loss_track = []

    # store encoder inputs [source morphological tags + target morphological tags + source word]
    source_data = []
    # store decoder expected outputs [target word]      
    target_data = []

    # stores encoded forms
    alphabet_and_morph_tags = dict()

    # read from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('exp_number')
    parser.add_argument('morph_tags_separator')
    parser.add_argument('source_morph_tag_number')
    parser.add_argument('target_morph_tag_number')
    args = parser.parse_args()

    source_data, target_data = data_functions.read_split_encode_data(args.filename, alphabet_and_morph_tags, parameters, args.morph_tags_separator, int(args.source_morph_tag_number), int(args.target_morph_tag_number))

    # run exp_number random parameterised experiments
    for exp_num in range(0,int(args.exp_number)):

        # generate parameters randomly
        parameters = parameters_class.experiment(source_data)
	# uncomment this line to train on the best parametercombination
        parameters = parameters_class.Parameters(2,1,0,10,300,256,100,0.0001,10,64,0.001)

        batch_size = parameters.batch_size

        # create trained_models directory if not existing, contains trained models
        data_functions.create_directory('./trained_models')

        # create parameters directory if not existing, contains trained models' parameters
        data_functions.create_directory('./parameters')

        # create output directory if not existing, contains Tensorboard outputs
        data_functions.create_directory('./output')

        # write parameters out to a file which filename is similar as trained_model name
        parameters_class.write_parameters_to_file(parameters, args, exp_num)

        # Clears the default graph stack and resets the global default graph.
        tf.reset_default_graph() 

        # initializes a tensorflow session
        with tf.Session() as sess:
            # get max value of encoded forms
            max_alphabet_and_morph_tags = alphabet_and_morph_tags[max(alphabet_and_morph_tags.items(), key=operator.itemgetter(1))[0]]

            # calculate vocab_size
            vocab_size = max_alphabet_and_morph_tags + 1

            # num neurons
            encoder_hidden_units = parameters.neuron_num 
            # in original paper, they used same number of neurons for both encoder
            # and decoder, but we use twice as many so decoded output is different
            decoder_hidden_units = encoder_hidden_units * 2 

            encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            # contains the lengths for each of the sequence in the batch, we will pad so all the same
            encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
            decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

            # used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
            # reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
            embeddings = tf.Variable(tf.eye(vocab_size, parameters.input_embedding_size), dtype='float32', name='embeddings')

            encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

            # define encoder
            encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

            # define bidirectionel function of encoder (backpropagation)
            ((encoder_fw_outputs,
            encoder_bw_outputs),
            (encoder_fw_final_state,
            encoder_bw_final_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
            )

            # Concatenates tensors along one dimension.
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            # because by GRUCells the state is a Tensor, not a Tuple like by LSTMCells
            encoder_final_state = tf.concat(
                (encoder_fw_final_state, encoder_bw_final_state), 1)

            # define decoder
            decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

            #we could print this, won't need
            encoder_max_time, parameters.batch_size = tf.unstack(tf.shape(encoder_inputs))
            # (character_changing_num-1) additional steps, +1 leading <EOS> token for decoder inputs
            decoder_lengths = encoder_inputs_length + parameters.character_changing_num

            #manually specifying since we are going to implement attention details for the decoder in a sec
            #weights
            W = tf.Variable(tf.eye(decoder_hidden_units, vocab_size), dtype='float32', name='W')
            #bias
            b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32, name='b')

            #create padded inputs for the decoder from the word embeddings
            #were telling the program to test a condition, and trigger an error if the condition is false.
            assert parameters.EOS == 1 and parameters.PAD == 0 and parameters.SOS == 2

            sos_time_slice = tf.fill([parameters.batch_size], 2, name='SOS')
            eos_time_slice = tf.ones([parameters.batch_size], dtype=tf.int32, name='EOS')
            pad_time_slice = tf.zeros([parameters.batch_size], dtype=tf.int32, name='PAD')

            # send batch size sequences into encoder at one time
            parameters.batch_size = batch_size

            #retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
            sos_step_embedded = tf.nn.embedding_lookup(embeddings, sos_time_slice)
            eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
            pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
    
            #manually specifying loop function through time - to get initial cell state and input to RNN
            
            #we define and return these values, no operations occur here
            def loop_fn_initial():
                initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                initial_input = sos_step_embedded
                #last time steps cell state
                initial_cell_state = encoder_final_state
                #none
                initial_cell_output = None
                # none
                initial_loop_state = None  # we don't need to pass any additional information
                return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)


            #attention mechanism --choose which previously generated token to pass as input in the next timestep
            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

                def get_next_input():
                    #dot product between previous ouput and weights, then + biases
                    output_logits = tf.add(tf.matmul(previous_output, W), b)
                    #Logits simply means that the function operates on the unscaled output of 
                    #earlier layers and that the relative scale to understand the units is linear. 
                    #It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
                    #(you might have an input of 5).
                    #prediction value at current time step
        
                    #Returns the index with the largest value across axes of a tensor.
                    prediction = tf.argmax(output_logits, axis=1)
                    #embed prediction for the next input
                    next_input = tf.nn.embedding_lookup(embeddings, prediction)
            
                    return next_input
    
                elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended
    
                #Computes the "logical and" of elements across dimensions of a tensor.
                finished = tf.reduce_all(elements_finished) # -> boolean scalar
                #Return either fn1() or fn2() based on the boolean predicate pred.
                input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    
                #set previous to current
                state = previous_state
                output = previous_output
                loop_state = None

                return (elements_finished, 
                    input,
                    state,
                    output,
                    loop_state)

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
		# time == 0
                if previous_state is None:
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

            #Creates an RNN specified by RNNCell cell and loop function loop_fn.
            #This function is a more primitive version of dynamic_rnn that provides more direct access to the 
            #inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
            #and what to emit for the output.
            #ta = tensor array
            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()

            #Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
            #reduces dimensionality
            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            #flettened output tensor
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
            #pass flattened tensor through decoder
            decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
            #prediction vals
            decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

            #final prediction
            decoder_prediction = tf.argmax(decoder_logits, 2)

            #cross entropy loss
            #one hot encode the target values so we don't rank just differentiate
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
                logits=decoder_logits,
            )

            #loss function
            loss = tf.reduce_mean(stepwise_cross_entropy)

            # tensorboard visualisation
            tf.summary.scalar('loss' + str(exp_num), loss)
            tf.summary.histogram('loss' + str(exp_num), loss)

            # if we use learning rate decay uncomment this line
            #learning_rate = tf.placeholder(tf.float32)
            learning_rate = parameters.learning_rate

            #train it 
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            sess.run(tf.global_variables_initializer())

            try:
                saver = tf.train.Saver()

                summ = tf.summary.merge_all()
                writer = tf.summary.FileWriter('output', sess.graph)

                train_model(source_data, target_data, encoder_inputs, encoder_inputs_length, decoder_targets, train_op, loss, decoder_prediction, sess, loss_track, parameters,saver, alphabet_and_morph_tags, exp_num, learning_rate, summ, writer)

            except KeyboardInterrupt:
                print('training interrupted')


    # write out vocab, because needed at inference and test scripts
    data_functions.write_vocab(alphabet_and_morph_tags)


if __name__ == '__main__':
    main()
