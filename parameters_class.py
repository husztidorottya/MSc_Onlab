import random


# class stores model parameters
class Parameters:
   def __init__(self, SOS, EOS, PAD, character_changing_num, input_embedding_size, neuron_num, epoch, delta, patience, batch_size, learning_rate):
      self.SOS = SOS
      self.EOS = EOS
      self.PAD = PAD
      self.character_changing_num = character_changing_num
      self.input_embedding_size = input_embedding_size
      self.neuron_num = neuron_num
      self.epoch = epoch
      self.early_stopping_delta = delta
      self.early_stopping_patience = patience
      self.batch_size = batch_size
      self.learning_rate = learning_rate


# we need to load the trained model parameters
def load_parameters(trained_model):
      parameters = Parameters(2, 1, 0, 10, 300, 100, 100, 0.001, 5, 20, 0.001)
      with open('parameters/' + trained_model + '_parameters.tsv', 'r') as input_parameters:
            line_num = 0
            for line in input_parameters:
                 param_line = line.strip('\n').split('\t')
                 if line_num == 0:
                     parameters.input_embedding_size = int(param_line[1])
                 if line_num == 1:
                     parameters.neuron_num = int(param_line[1])
                 if line_num == 2:
                     parameters.epoch = int(param_line[1])
                 if line_num == 3:
                     parameters.early_stopping_delta = float(param_line[1])
                 if line_num == 4:
                     parameters.early_stopping_patience = int(param_line[1])
                 if line_num == 5:
                     parameters.batch_size = param_line[1]
                 if line_num == 6:
                     parameters.learning_rate = float(param_line[1])
                 line_num += 1

      return parameters


# create random parameterscombination from hyperparameter space
def experiment(source_data):
    # hyperparameter space
    neuron_num_param = [20, 32, 50, 64, 100, 128, 200, 256]
    input_embedding_size_param = [32, 50, 100, 256, 300]
    batch_size_param = [20, 32, 64]
    epoch_param = [10, 20, 30, 40, 100]
    early_stopping_patience_param = [3, 5, 10]
    early_stopping_delta_param = [0.001, 0.0001]
    learning_rate_param = [0.1, 0.01, 0.001]

    return Parameters(2, 1, 0, 10, random.choice(input_embedding_size_param), random.choice(neuron_num_param), random.choice(epoch_param), random.choice(early_stopping_delta_param), random.choice(early_stopping_patience_param), random.choice(batch_size_param), random.choice(learning_rate_param))


# log model's parameters
def write_parameters_to_file(parameters, args, exp_num):
    # write out parameters (we need it at test_accuracy and inferenc, because the models are different when parameters different)
    with open('parameters/' + 'trained_model'+ str(exp_num) + '_parameters.tsv', 'w') as output_parameters:
        output_parameters.write('input_embedding_size\t{}\n'.format(parameters.input_embedding_size))
        output_parameters.write('neuron_num\t{}\n'.format(parameters.neuron_num))
        output_parameters.write('epoch\t{}\n'.format(parameters.epoch))
        output_parameters.write('early_stopping_delta\t{}\n'.format(parameters.early_stopping_delta))
        output_parameters.write('early_stopping_patience\t{}\n'.format(parameters.early_stopping_patience))
        output_parameters.write('batch_size\t{}\n'.format(parameters.batch_size))
        output_parameters.write('learning_rate\t{}\n'.format(parameters.learning_rate))

    return



