import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import LongTensor, Tensor


class BertScreenVocab(object):
    vocab_list: [str]

    def __init__(self, vocab_list: [str], vocab_size: int, bert_model: SentenceTransformer, bert_size: object = 768,
                 embedding_path: str = None):
        """
        vocab_list : list of all possible text labels on screens in dataset
        vocab_size : length of vocab_list
        bert_model : sentence BERT model to encode text
        bert_size : the length of bert_model's embeddings
        """
        self.vocab_list = vocab_list
        self.vocab_list.append('')
        self.bert = bert_model
        self.embeddings = self.load_embeddings(embedding_path)
        self.text_to_index = {}
        self.load_indices()
        self.bert_size = bert_size

    def load_indices(self):
        for index in range(len(self.vocab_list)):
            self.text_to_index[self.vocab_list[index]] = index

    def load_embeddings(self, embedding_path: str) -> Tensor:
        if embedding_path:
            vocab_emb = np.load(embedding_path)
            empty_emb = self.bert.encode([''])
            vocab_emb = np.concatenate((vocab_emb, empty_emb), axis=0)
        else:
            vocab_emb = self.bert.encode(self.vocab_list)

        return torch.as_tensor(vocab_emb)

    def get_index(self, text: str) -> LongTensor:
        """
        given a vector of text labels, identifies the index at which each 
        (and its embedding) is located, returns those indices as a tensor
        """
        vec: LongTensor = torch.LongTensor(len(text))
        for index in range(len(vec)):
            vec[index] = self.text_to_index[text[index]]
        return vec

    def get_text(self, index: int) -> str:
        """
        returns the text found at index
        """
        return self.vocab_list[index]

    def get_embedding_for_cosine(self, index: int):
        """
        returns the embedding at index
        """
        # index is an integer
        emb = self.embeddings[index]
        return emb

    def get_embeddings_for_softmax(self, index: int):
        """
        takes in a tensor of indices, returns the embeddings at those indices
        """
        # index is a tensor
        result_embeddings = self.embeddings.gather(0, index)
        return result_embeddings
