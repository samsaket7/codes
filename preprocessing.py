import numpy as np

#PREPROCESSING::

char_list = np.asarray([ord(char) for char in open('test.txt').read()])

sequence_length = 5
num_chars = np.size(char_list)
num_data = int(num_chars/sequence_length)
data_dim = 8

char_binary_list = ((char_list[:,None] & (1 << np.arange(data_dim))) > 0).astype(int)

rc = np.random.randint(0,num_chars-sequence_length-1,num_data)

input_data = np.asarray([char_binary_list[rc+i] for i in range(sequence_length)])
output_data = np.asarray([char_binary_list[rc+i] for i in range(1,sequence_length+1)])

assert char_list.size == num_chars, "char_list_size is wrong"
assert char_binary_list.shape == (num_chars,data_dim), "Error while converting ASCII into binary"
assert input_data.shape == (sequence_length,num_data,data_dim), "Shape of input data is wrong" 
assert output_data.shape == (sequence_length,num_data,data_dim), "Shape of output data is wrong"

#POSTPROCESSING::

y = output_data.dot(1<<np.arange(data_dim))
print(y)
print(y.astype(np.uint8).view('S1'))
