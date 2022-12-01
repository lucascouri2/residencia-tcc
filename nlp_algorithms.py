import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from scipy import spatial

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

from torch.utils.data import DataLoader

SEED = 42

'''import pandas as pd
import numpy as np
import time

from sklearn.metrics.pairwise import cosine_similarity

from scipy import spatial

from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

SEED = 42'''


def join_docs(df_text, column1, column2, unique = False):

    data = []
    for i in range(0, len(df_text)):
        data.append(' '.join(df_text[column1][i]))
        data.append(' '.join(df_text[column2][i]))

    if unique == True:
        data = list(set(data))

    return data



###Bag of Words###
def get_bow(doc1, doc2):
    
    vectorizer = CountVectorizer()
    
    text_list1 = ' '.join(doc1)
    text_list2 = ' '.join(doc2)
    
    text_list = [text_list1, text_list2]

    vector = vectorizer.fit_transform(text_list)
    
    cosine_similarities = cosine_similarity(vector[0], vector[1])
    
    return cosine_similarities[0][0]


def apply_bow(df, len_pipeline):

    time_list = []
    df_bow = pd.DataFrame()
    for index in range(len_pipeline):    
        start_time = time.time()
        df_bow[f'bow{index}'] = df.apply(lambda row: get_bow(row[f'doc1_pipeline{index}'], row[f'doc2_pipeline{index}']), axis=1)
        time_list.append((f'bow{index}', time.time()-start_time))

    return (df_bow, pd.DataFrame(time_list))



###TF-IDF###
def calculate_tf_idf(df, column1, column2):
    
    data = join_docs(df, column1, column2)
      
    tf_idf = TfidfVectorizer().fit_transform(data)
    
    return tf_idf


def get_tf_idf(tf_idf, index):
    
    index1 = 2*index
    index2 = 2*index + 1
    
    cosine_similarities = cosine_similarity(tf_idf[index1], tf_idf[index2])
    
    return cosine_similarities[0][0]


def apply_tf_idf(df, len_pipeline):
    
    time_list = []
    df_tf_idf = pd.DataFrame()

    for index in range(len_pipeline):
        
        start_time = time.time()
        tf_idf = calculate_tf_idf(df, f'doc1_pipeline{index}', f'doc2_pipeline{index}')

        tf_idf_list = []
        for index_df in range(len(df)):
            
            tf_idf_list.append(get_tf_idf(tf_idf, index_df))
        
        df_tf_idf[f'tf_idf{index}'] = tf_idf_list
        time_list.append((f'tf_idf{index}', time.time()-start_time))
    
    return (df_tf_idf, pd.DataFrame(time_list))



###Bert###
'''def get_bert(model, doc1, doc2):
    
    data = [doc1, doc2]
    sentence_embeddings = model.encode(data)

    infer1 = sentence_embeddings[0]
    infer2 = sentence_embeddings[1]
    
    cos_similarity = 1 - spatial.distance.cosine(infer1, infer2) #de 0 a 1
    
    return cos_similarity


def apply_bert(df, len_pipeline, model, model_name):

    time_list = []

    df_bert = pd.DataFrame()
    for index in range(len_pipeline):
        start_time = time.time()
        df_bert[f'bert_{model_name}{index}'] = df.apply(lambda row: get_bert(model, " ".join(row[f'doc1_pipeline{index}']), " ".join(row[f'doc2_pipeline{index}'])), axis=1)
        time_list.append((f'bert_{model_name}{index}', time.time()-start_time))

    return (df_bert, pd.DataFrame(time_list))'''

###Bert###
def get_bert(model, doc1, doc2):
    
    data = [doc1, doc2]
    sentence_embeddings = model.encode(data)

    infer1 = sentence_embeddings[0]
    infer2 = sentence_embeddings[1]
    
    cos_similarity = 1 - spatial.distance.cosine(infer1, infer2) #de 0 a 1
    
    return cos_similarity

def apply_bert(df, df_test, len_pipeline, model, model_name, fine_tune = False):

    train_batch_size = 128
    num_epochs = 2
    model_save_path = 'output/fine_tune_bert-'+model_name

    time_list = []

    df_bert = pd.DataFrame()
    for index in range(len_pipeline):
        start_time = time.time()
        if fine_tune==True:

            train_samples = []
            valid_samples = []

            X = df[[f'doc1_pipeline{index}', f'doc2_pipeline{index}']]
            y = df['Overall']

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = SEED)

            train_df = pd.concat([X_train, y_train], axis=1)
            train_df = train_df.reset_index()
            valid_df = pd.concat([X_valid, y_valid], axis=1)
            valid_df = valid_df.reset_index()


            ### testar sem tokenizar
            for i in range(1, len(train_df)):
                inp_example = InputExample(texts=[train_df.iloc[i][f'doc1_pipeline{index}'], train_df.iloc[i][f'doc2_pipeline{index}']], label=float(train_df.iloc[i]['Overall']))
                train_samples.append(inp_example)

            for j in range(1, len(valid_df)):
                inp_example = InputExample(texts=[valid_df.iloc[j][f'doc1_pipeline{index}'], valid_df.iloc[j][f'doc2_pipeline{index}']], label=float(valid_df.iloc[j]['Overall']))
                valid_samples.append(inp_example)

            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)
            evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(valid_samples)#(sts_reader.get_examples('sts-dev.csv'))

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                    evaluator=evaluator,
                    epochs=num_epochs,
                    #evaluation_steps=1000,
                    #warmup_steps = 100,
                    output_path=model_save_path
                    )

        df_bert[f'bert_{model_name}{index}'] = df_test.apply(lambda row: get_bert(model, " ".join(row[f'doc1_pipeline{index}']), " ".join(row[f'doc2_pipeline{index}'])), axis=1)
        time_list.append((f'bert_{model_name}{index}', time.time()-start_time))

    return (df_bert, pd.DataFrame(time_list))


###Doc2Vec###
def train_doc2vec(df, len_pipeline):

    for index in range(len_pipeline):
        data = join_docs(df, f'doc1_pipeline{index}', f'doc2_pipeline{index}', unique = True)
        tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)] 
        
        model = gensim.models.doc2vec.Doc2Vec(vector_size = 30, min_count = 0, epochs = 80)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples = model.corpus_count, epochs = 80)
        model.save(f'd2v{index}.model')


def get_doc2vec(model, doc1, doc2):
    
    infer1 = model.infer_vector(doc1)
    infer2 = model.infer_vector(doc2)
    
    cos_similarity = 1 - spatial.distance.cosine(infer1, infer2) #de 0 a 1
    
    return cos_similarity
    
    
def apply_doc2vec(df, len_pipeline):
    
    time_list = []    
    df_doc2vec = pd.DataFrame()    
    for index in range(len_pipeline):
        start_time = time.time()
        model = Doc2Vec.load(f'd2v{index}.model')
        df_doc2vec[f'doc2vec{index}'] = df.apply(lambda row: get_doc2vec(model, (row[f'doc1_pipeline{index}']), (row[f'doc2_pipeline{index}'])), axis=1)
        time_list.append((f'doc2vec{index}', time.time()-start_time))

    return (df_doc2vec, pd.DataFrame(time_list))



###Word2Vec###
def get_mean_vector(model, words): #words é um documento inteiro

    # remove out-of-vocabulary words
    words = [word for word in words if word in model.index_to_key]

    if len(words) >= 1:
        return np.mean(model[words], axis = 0)
    else:
        return []


def get_word2vec(model, doc1, doc2):
    
    infer1 = get_mean_vector(model, doc1)
    infer2 = get_mean_vector(model, doc2)
    
    cos_similarity = 1 - spatial.distance.cosine(infer1, infer2) #de 0 a 1
    
    return cos_similarity


def apply_word2vec(df, len_pipeline, model):
    
    time_list = [] 
    df_word2vec = pd.DataFrame()
    for index in range(len_pipeline):
        start_time = time.time()
        df_word2vec[f'word2vec{index}'] = df.apply(lambda row: get_word2vec(model, row[f'doc1_pipeline{index}'], row[f'doc2_pipeline{index}']), axis=1)
        time_list.append((f'word2vec{index}', time.time()-start_time))

    return (df_word2vec, pd.DataFrame(time_list))
