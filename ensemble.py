import os 
import random
import numpy as np
import joblib as jb
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from parser_ensemble import parameter_parser

# file to byte vector
def byte_occurence(file_path):
    with open(file_path , 'rb') as file:
        binary_data = file.read()
    seq_vectorizer = jb.load('./byte_vectorizer.joblib')
    seq = ''
    for byte in binary_data:
        seq += (str(byte) + ' ')
    result = seq_vectorizer.transform([seq])
    result = np.array(result).tolist().astype(float)
    return(result)

# byte occurence predict
def byte_occurence_model_predict(opcode_occurence):
    occurence_model = jb.load('./byte_frequency_model.joblib')
    result = occurence_model.predict(opcode_occurence)
    return result[0]

# the function to help calculating entropy 
def calculate_entropy(file_path):
    # 计算给定数据的熵
    length = len(data)
    frequencies = Counter(data)
    probabilities = [count / length for count in frequencies.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_byte_frequency(byte_sequence):
    # 使用 Counter 统计字节出现的次数
    byte_counter = Counter(byte_sequence)
    byte_frequency = [byte_counter[i] for i in range(256)]
    return byte_frequency

# file to byte netropy vector
def byte_entropy(file_path):
    with open(file_path , 'rb') as file:
        binary_data = file.read()
    sequence = ''
    result = np.zeros(256,dtype=int).tolist()
    window_shape = 4096
    step_size = 2048
    running_seat = 0
    window_number = 0
    if len(binary_data) >= 4096:
        while((running_seat + 4096) <= len(binary_data)):
            byte_occurence = calculate_byte_frequency(binary_data[running_seat : (running_seat+4096)])
            length = np.linalg.norm(np.array(byte_occurence))
            byte_occurence = [x/length for x in byte_occurence]
            result = [result[i] + byte_occurence[i] for i in range(256)]
            running_seat += 2048
            window_number += 1
    else:
        byte_occurence = calculate_byte_frequency(binary_data)
        length = np.linalg.norm(np.array(byte_occurence))
        byte_occurence = [x/length for x in byte_occurence]
        result = [result[i] + byte_occurence[i] for i in range(256)]
        window_number += 1
    result = [entropy / window_number for entropy in result]
    return [result]

# file entropy predict
def byte_entropy_model_predict(opcode_entropy):
    entropy_model = jb.load('./byte_entropy_model.joblib')
    result = entropy_model.predict(opcode_entropy)
    return result[0]

# extract opcode sequence
def file_to_opcode(file_path):
    cmd = '../../retdec-install/bin/retdec-decompiler -output '
    cmd += (file_path.split('/')[-1] + ' --silent ' + file_path)
    os.system(cmd)
    try:
        os.system('rm ./' + file_path.split('/')[-1])
        os.system('rm ./' + file_path.split('/')[-1] + '.bc')
        os.system('rm ./' + file_path.split('/')[-1] + '.config.json')
        os.system('rm ./' + file_path.split('/')[-1] + '.ll')
        f = open('./' + file_path.split('/')[-1] + '.dsm' , 'r')
        os.system('rm ./' + file_path.split('/')[-1] + '.dsm')
        op = ''
        for j in f.readlines():
            if('\t' in j):
                j = j.split('\t')[1].split(' ')[0]
                op += (j + ' ') 
        # os.system('rm ./' + file_path.split('/')[-1] + '.dsm')
    except:
        return False
    return op

# opcode sequence to vector
def opcode_seq_to_vector(opcode_seq):
    if len(opcode_seq.split(' ')) <= 2:
        return np.zeros((1,1000))
    op_vectorizer = CountVectorizer(ngram_range=(2,4))
    x_op_count = op_vectorizer.fit_transform([opcode_seq])
    top_features = np.load('./top_features.npy').tolist()
    feature_list = op_vectorizer.get_feature_names()
    top_features_index = []
    for i in range(len(top_features)):
        try:
            top_features_index.append(feature_list.index(top_features[i]))
        except:
            top_features_index.append(-1)
    X_op_fre = []

    vector = []
    for j in range(len(top_features_index)):
        if top_features_index[j] == -1:
            vector.append(0)
        else:   
            vector.append(x_op_count[:, top_features_index[j]][0, 0])
    return np.array(vector).reshape(1,1000)

# opcode predict
def opcode_fre_model_predict(opcode_vector):
    opcode_model = jb.load('./xgboost')
    result = opcode_model.predict(opcode_vector)
    return result[0]

def generate_layer2_input(file_path):
    vector = []
    # byte occurence
    tmp = byte_occurence(file_path)
    vector.append(byte_occurence_model_predict(tmp))
    
    # byte entropy
    tmp = byte_entropy(file_path)
    vector.append(byte_entropy_model_predict(tmp))
    
    # opcode
    opcode_seq = file_to_opcode(file_path)
    if opcode_seq:
        tmp = opcode_seq_to_vector(opcode_seq)
        vector.append(opcode_fre_model_predict(tmp))
    else:
        return 'F'
    
    return np.array(vector)[np.newaxis,:]

def predict_result(vector):
    model = jb.load('./ensemble_model_30000.joblib')
    result = model.predict(vector)
    return result[0]

def main(args):
    tmp = generate_layer2_input(args.input_path)
    if type(tmp) != str:
        result = predict_result(tmp)
    else:
        print('Fail to extract feature')
        return 'Fail to extract feature'
    if result == 0:
        print('Benignware')
        return 'Benignware'
    if result == 1:
        print('Malware')
        return 'Malware'
    
if __name__=='__main__':
    args = parameter_parser()
    main(args)
    