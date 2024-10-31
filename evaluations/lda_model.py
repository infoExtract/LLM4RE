import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
import nltk
import joblib
import pickle
import os
import json
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and word.isalnum()]


def main(args):
    print(f"Training LDA model for {args.task}...")

    outpath = f'{args.output_dir}/{args.task}'
    os.makedirs(outpath, exist_ok=True)

    all_source_texts = pd.read_json(f'{args.data_dir}/{args.task}/test.json', lines=True)['text'].to_list()

    # Preprocess all texts
    processed_texts = [preprocess(text) for text in all_source_texts]

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    with open(f'{outpath}/dictionary.joblib', 'wb') as f:
        joblib.dump(dictionary, f)

    # Train LDA model
    lda = LdaModel(corpus, num_topics=args.topic_nums, id2word=dictionary)

    with open(f'{outpath}/lda.joblib', 'wb') as f:
        joblib.dump(lda, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, required=False, help="Dataset Name.")
    parser.add_argument('--topic_nums', type=int, default=150, required=False,
                        help="number of topics for LDA")

    parser.add_argument('--data_dir', '-dir', type=str, required=False,
                        default="/home/UFAD/aswarup/research/Relation-Extraction/ICL/Data_ICL/data_rc")
    parser.add_argument("--output_dir", default='./output', type=str, required=False,
                        help="The output directory where the lda model")

    args = parser.parse_args()
    main(args)
