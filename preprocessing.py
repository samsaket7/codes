import numpy as np

def str_2_asciiVec(string):

	ascii_list    = np.asarray([ord(char) for char in string])
	asciiVec_list = ((ascii_list[:,None] & (1 << np.arange(8))) > 0).astype(int)
	assert asciiVec_list.shape == (len(string),8) , "Error in str_2_asciiVec"
	return asciiVec_list


def asciiVec_2_str(asciiVec_list):
	
	assert asciiVec_list.ndim == 2 and asciiVec_list.shape[1] == 8, "Wrong input shape to asccVec_2_str"
	ascii_list = asciiVec_list.dot(1<<np.arange(8))
	string = "".join([chr(asci) for asci in ascii_list])
	return string


def preprocess(path_of_file='input.txt',sequence_length=50):
	string = open(path_of_file).read()
	asciiVec_list = str_2_asciiVec(string)
	num_data = int(len(string)/sequence_length)

	rc = np.random.randint(0,len(string)-sequence_length-1,num_data)
	input_data = np.asarray([asciiVec_list[rc+i] for i in range(sequence_length)])
	output_data = np.asarray([asciiVec_list[rc+i] for i in range(1,sequence_length+1)])

	assert input_data.shape == (sequence_length,num_data,8), "Shape of input data is wrong" 
	assert output_data.shape == (sequence_length,num_data,8), "Shape of output data is wrong"

	return input_data,output_data
    
def postprocess(predicted_data,path_of_file='output.txt'):
	assert predicted_data.ndim == 3 and predicted_data.shape[2] == 8, "Wrong input shape to asccVec_2_str"
	sequence_length,num_data,_ = predicted_data.shape
	output_file = open(path_of_file,'w')
	for i in range(num_data):
		string = asciiVec_2_str(predicted_data[:,i,:])
		output_file.write('Data::'+str(i)+'   '+string+'\n\n\n\n')

	
input_data,output_data = preprocess()    
postprocess(output_data)

