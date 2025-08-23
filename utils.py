import os

def stream_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            yield line

def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r') as f:
        sentences = [[word for word in line.strip().split()] for line in f]
        print(sentences[0:10])
        return sentences


def read_tokenized_data(tokenized_path):
    if not os.path.exists(tokenized_path):
        raise FileNotFoundError
    
    with open(tokenized_path, 'r') as f:
        tokens = [int(token)  for line in f for token in line.strip().split()]
        return tokens