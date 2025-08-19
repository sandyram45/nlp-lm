from collections import defaultdict, deque
from utils.data_utils import read_data
class Ngrams:
    def __init__(self,N, optimized=False):
        self.n = N
        # dictionary of set as key and dict as values where this dict will have words : count
        self.word_dict = defaultdict(lambda: defaultdict(int)) if optimized else dict()
        self.vocab = set()
        self.optimized = optimized

    def preprocess_text(self,filepath):
        data = []
        for sent in read_data(filepath):
            data.append(sent)
            for word in sent:
                if isinstance(word, str):
                    self.vocab.add(word)
        return [['<START>'] * (self.n - 1) + sentence + ['<END>'] for sentence in data]

    def generate_text(self,text):
        pass

    def get_probabilities(self,text,smoothing=''):
        if not isinstance(smoothing, str):
            raise TypeError("Smoothing variable must of type str")
        if isinstance(text, list):
             text_split = text
        elif isinstance(text, str): 
            text_split = text.split()
        else:
             raise TypeError('Text must be a string or a list of tokens')
        context = tuple(text_split[len(text_split) - self.n + 1: ])
        find_word = self.word_dict.get(context, {})
        total_count = sum(find_word.values())

        if smoothing == 'laplace':
             return { word : ( find_word.get(word, 0) + 1) / (total_count + len(self.vocabulary)) for word in self.vocabulary}
        
        if smoothing == 'kneser-ney':
            pass
        
        if smoothing == 'modified-kneser-ney':
            pass

        return { key : value/total_count for key,value in find_word.items()} if total_count > 0 else {}
        
    def __str__(self):
        return f"{self.n}-grams Language Model"
    def __repr__(self):
        return f"{self.n}-grams Language Model"