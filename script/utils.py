from zeugma.embeddings import EmbeddingTransformer
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
# from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from torchtext.vocab import FastText
import torch
import numpy as np
import nltk

from gpt_utils import init_gpt

nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
nltk.download('punkt')
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# init models

# BERT
bert = 'bert-base-uncased'
bert_model = SentenceTransformer(bert)

# tfidf
tf = TfidfVectorizer(max_features=768)

# bow
bow_vect = CountVectorizer(
    stop_words='english', ngram_range=(2, 3), max_features=768)

# lda
lda = LatentDirichletAllocation()

# lsa
lsa = TruncatedSVD()

# fastText
embedding_fastText = FastText('simple')

# glove
glove = EmbeddingTransformer('glove')


# fit some vect
def fit_init(sents):
    tf.fit(sents)
    bow_vect.fit(sents)
    return True

# embedding extractors


# semantic relation
def sr_vectorizer(sentences):
    def vect_doc(doc, model):
        vectors = []
        vectors = [model.wv[token] for token in doc if token in model.wv]
        if vectors:
            return np.asarray(vectors).mean(axis=0)
        return np.zeros(model.vector_size)

    model = Word2Vec(sentences=sentences, size=768,
                     workers=1, window=15, seed=42)

    return np.array([vect_doc(doc, model) for doc in sentences])


def lda_transform(item):

    return lda.fit_transform(tf.transform(item['tweet']).A, item['Y'].to_numpy())


def lsa_transform(item):

    return lsa.fit_transform(tf.transform(item['tweet']).A, item['Y'].to_numpy())

# trans function


# bert
embedd_fn = FunctionTransformer(lambda item: bert_model.encode(
    item, convert_to_tensor=True, show_progress_bar=True).detach().cpu().numpy())

# tfidf
tf_idf_fn = FunctionTransformer(lambda item: tf.transform(item).A)

# lda
lda_fn = FunctionTransformer(lambda item: lda_transform(item))

# lsa
lsa_fn = FunctionTransformer(lambda item: lsa_transform(item))

# glove
glove_fn = FunctionTransformer(lambda item: glove.transform(item))

# bow
bow_vect_fn = FunctionTransformer(lambda item: bow_vect.transform(item).A)

# semantic_rel
semantic_rel_fn = FunctionTransformer(lambda item: sr_vectorizer(item))

# fastText
fastText_fn = FunctionTransformer(lambda item: np.array([torch.mean(
    embedding_fastText.get_vecs_by_tokens(x), dim=0).tolist() for x in item]))

# gpt
gpt_fn = FunctionTransformer(lambda item: init_gpt(item))
