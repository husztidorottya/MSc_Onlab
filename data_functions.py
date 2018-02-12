import os

# handle tasks as input
def encoding_input(data_line_, item, alphabet_and_morph_tags, separator):

    if data_line_[item].find(separator) != -1:
        # split morphological tags
        tags = data_line_[item].split(separator)
        coded_word = encoding(tags, alphabet_and_morph_tags)

    else:
        # encode source and target word
        coded_word = encoding(data_line_[int(item)],  alphabet_and_morph_tags)

    return coded_word


# create sequence for input or output
def create_sequence(data_line_, issource, parameters):
    sequence = []

    # source sequence
    if issource:
        # do not count the target word in the length
        for j in range(0,len(data_line_)-1):
            # if it hits to the source word SOS character is appended
            if j == len(data_line_)-2:
                sequence.append(parameters.SOS)
            # appending the next element of the array (source morph tags / target morph tags / source word)
            for i in data_line_[j]:
                sequence.append(i)
            # if it hits to the source word EOS character is appended
            if j == len(data_line_)-2:
                sequence.append(parameters.EOS) 
    # target sequence
    else:
        sequence.append(parameters.SOS)
        
        if len(data_line_) == 4:
            # append target word
            for i in data_line_[3]:
                sequence.append(i)
        else:
            for i in data_line_[2]:
                sequence.append(i)
             
    return sequence


# encoding characters
def encoding(data, alphabet_and_morph_tags):
    coded_word = []
    for character in data:
        index = alphabet_and_morph_tags.setdefault(character, len(alphabet_and_morph_tags) + 3)
        coded_word.append(index)
        
    return coded_word


# read, split and encode input data
def read_split_encode_data(filename, alphabet_and_morph_tags, parameters, separator, source_morph_tag_number, target_morph_tag_number):
    with open(filename,'r') as input_file:
        source_data = []
        target_data = []
        idx = 0
        # read it line-by-line
        for line in input_file:
            data_line_ = line.strip('\n').split('\t')
            data_line = []

            # source_moprh_tag_number stores the source morph tags' position in line, put it its place
            if source_morph_tag_number != -1:
                coded_word = encoding_input(data_line_, source_morph_tag_number, alphabet_and_morph_tags, separator)

                data_line.append(coded_word)

            # target_morph_tag_number stores the target morph tags' position in line, put it its place
            if target_morph_tag_number != -1:
                coded_word = encoding_input(data_line_, target_morph_tag_number, alphabet_and_morph_tags, separator)
                
                data_line.append(coded_word)
                
            # encode words into vector of ints 
            for item in range(0,len(data_line_)):         
                if item != source_morph_tag_number and item != target_morph_tag_number:

                    # contains encoded form of word
                    coded_word = []
            
                    coded_word = encoding_input(data_line_, item, alphabet_and_morph_tags, separator)

                    # store encoded form
                    data_line.append(coded_word)

            # store encoder input - morph tags + source word
            source_data.append([create_sequence(data_line, True, parameters), idx])

            # store decoder expected outputs -target word
            target_data.append(create_sequence(data_line, False, parameters))
            
            # stores line number (needed for shuffle) - reference for the target_data
            idx += 1

    return source_data, target_data


# read vocab
def read_vocab(alphabet_and_morph_tags):
    with open('alphabet_and_morph_tags.tsv','r') as inputfile:
          for line in inputfile:
               line_data = line.strip('\n').split('\t')
               alphabet_and_morph_tags[line_data[0]] = int(line_data[1])

    return alphabet_and_morph_tags


# write out vocab, because needed at inference and test scripts
def write_vocab(alphabet_and_morph_tags):
    with open('alphabet_and_morph_tags.tsv','w') as outputfile:
        for k,v in alphabet_and_morph_tags.items():
            outputfile.write('{}\t{}\n'.format(k,v))

    return 


# create directory for structuring data
def create_directory(directory):
    # create directory if not existing
    dir = directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    return


# converts vector of numbers back to characters
def convert_back_tostring(parameters, data, alphabet_and_morph_tags):
    word = ''
    for element in data:
         for char in element:
              if char != parameters.EOS and char != parameters.SOS and char != parameters.PAD:
                  # https://stackoverflow.com/questions/23295315/get-key-by-value-dict-python
                  word = word + (list(alphabet_and_morph_tags.keys())[list(alphabet_and_morph_tags.values()).index(char)])

    return word

