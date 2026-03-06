import os
import random
import json
import functools
import pickle

from gensim.models import Word2Vec, KeyedVectors
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from tqdm import tqdm


def build_word2vec(
    df,
    word2vec_cols,
    embedding_size=10,
    save=True,
    output_dir='./',
    force_rebuild=False
):
    """
    Train Word2Vec on the port and protocols together. Save (to disk) and return a KeyedVector object, a mapping
    object between raw "words" (protocols/ports as strings) and their embeddings (np.array). 
    
    Also returns a bool that signals whether the model was trained from scratch. This is needed because we need
    to rebuild the ANN and NN files if the model was retrained.
    """
    model_path = os.path.join(output_dir, f'word2vec_dim{embedding_size}.wv')

    rebuilt = False

    # Preload if possible. Otherwise train from scratch.
    if os.path.exists(model_path) and not force_rebuild:
        wv = KeyedVectors.load(model_path)
        print(f"Loaded Word2Vec from: {model_path}")
    else:
        # Construct the training corpus. Each "sentence" = a list of features from each row of the input data.
        # Only use columns that are actually in the input data from the ones specified.
        shared_cols = list(set(df.columns) & set(word2vec_cols))
        sentences = df.loc[:, shared_cols].astype(str).values.tolist()

        print(f'Training word2vec on: {shared_cols}')
        model = Word2Vec(
            sentences=sentences,
            vector_size=embedding_size,
            window=5,
            min_count=1
        )
        wv = model.wv

        if save:
            print(f'Word2Vec embeddings (KeyedVectors) saved here: {model_path}')
            wv.save(model_path)

        rebuilt = True

    return wv, rebuilt


def build_ann_index(
    df,
    word2vec: Word2Vec | KeyedVectors | str,
    col_type,
    target_cols,
    embedding_size=10,
    ann_n_trees=100,
    save=True,
    output_dir='./',
    rebuild=False
):
    """
    Creates an approximate nearest neighbor (ANN) index on the Word2Vec embeddings for the selected columns. 
    Because the ANN returns the indices of the vectors, also returns a dictionary mapping from these
    ANN indices back to the original "words".

    Used to map synthetically generated data, which may not map exactly to a viable "word" (port, protocol, etc).
    """

    # Reload existing ANN index if possible. Should not proceed upon failure (indicates retraining needed).
    if not rebuild:
        ann, idx_to_vocab = load_ann_index(
            col_type=col_type,
            embedding_size=embedding_size,
            output_dir=output_dir
        )
        return ann, idx_to_vocab

    # Get the word2vec mapping, either directly or from a file.
    if isinstance(word2vec, Word2Vec):
        wv = word2vec.wv
    elif isinstance(word2vec, KeyedVectors):
        wv = word2vec
    else:
        wv = KeyedVectors.load(word2vec, mmap='r')

    # Add embedding of each unique item to index.
    shared_cols = list(set(df.columns) & set(target_cols))
    vocab = functools.reduce(np.union1d, [df[col].unique() for col in shared_cols]).tolist()
    ann = AnnoyIndex(embedding_size, 'angular')
    idx_to_vocab = {}


    print(f'ANN Columns: {shared_cols}')
    print('Adding vectors to ANN index...')
    for i, elt in tqdm(enumerate(vocab)):
        try:
            vec = wv.get_vector(str(elt), norm=False)
            ann.add_item(i, vec)
            idx_to_vocab[i] = elt
        except KeyError:
            continue
    print('Building trees...')
    ann.build(ann_n_trees)

    # Save the ANN and mapping from ANN indices to original ports/protocol (vocab).
    if save:
        save_ann_index(
            ann=ann, 
            idx_to_vocab=idx_to_vocab, 
            col_type=col_type,
            output_dir=output_dir
        )

    return ann, idx_to_vocab


def save_ann_index(
    ann,
    idx_to_vocab,
    col_type,
    embedding_size=10,
    ann_n_trees=100,
    output_dir='./'
):
    """
    Saves ANN index and ANN vector indices mapping.
    """
    name = f'{col_type}_dim{embedding_size}_trees{ann_n_trees}'

    ann_path = os.path.join(output_dir, f'{name}_ann.ann')
    idx_dict_path = os.path.join(output_dir, f'{name}_idx_map.json')

    print(f'Saving ANN here: {ann_path}')
    print(f'Saving index to vocab dictionary here: {idx_dict_path}')

    ann.save(ann_path)
    with open(idx_dict_path, mode='w') as file:
        json.dump(idx_to_vocab, file)


def load_ann_index(
    col_type,
    embedding_size=10,
    ann_n_trees=100,
    output_dir='./'
    ):
    """
    Load saved ANN index and dictionary mapping ANN vector indices to vocab.
    """
    name = f'{col_type}_dim{embedding_size}_trees{ann_n_trees}'

    ann_path = os.path.join(output_dir, f'{name}_ann.ann')
    idx_dict_path = os.path.join(output_dir, f'{name}_idx_map.json')

    ann = AnnoyIndex(embedding_size, 'angular')
    ann.load(ann_path)

    with open(idx_dict_path) as file:
        # Have to convert keys to int (since JSON only allows str keys).
        raw = json.load(file)
        idx_to_vocab = {int(idx): original for idx, original in raw.items()}

    # print(f'Loaded ANN from: {ann_path}')
    # print(f'Loaded vector index mapping from: {idx_dict_path}')

    return ann, idx_to_vocab


def build_nn(word2vec: Word2Vec | KeyedVectors | str, output_dir='./', save=True, rebuild=False):
    """
    Build the nearest neighbor (NN) model used during the embedding creation phase
    to map out of vocabulary numeric words (ports/protocols) to closest valid words.
    """

    # Load pretrained NN if possible. Should not proceed upon failure (indicates retraining needed).
    if not rebuild:
        try:
            nn = load_nn(output_dir=output_dir)
            return nn
        except:
            pass


    # Get the word2vec mapping, either directly or from a file.
    if isinstance(word2vec, Word2Vec):
        wv = word2vec.wv
    elif isinstance(word2vec, KeyedVectors):
        wv = word2vec
    else:
        wv = KeyedVectors.load(word2vec, mmap='r')

    # NOTE: Assuming that underlying words are numeric. Should be fine since
    # we're working with port/protocol numbers...
    vocab = set(int(key) for key in wv.key_to_index)
    vocab_list = list(vocab)

    # Fit vocab to nearest neighbors.
    print('Building NN for Word2Vec columns...')
    vocab_arr = np.array(vocab_list).reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vocab_arr)

    if save:
        save_nn(nn, output_dir=output_dir)

    return nn


def save_nn(nn, output_dir='./'):
    """
    Saves the NN model used when transforming features to embeddings.
    """
    nn_path = os.path.join(output_dir, 'port2vec_nn.pkl')

    with open(nn_path, mode='wb') as file:
        pickle.dump(nn, file)

    print(f'Saved NN at: {nn_path}')


def load_nn(output_dir='./'):
    """
    Load the NN model used when transforming features to embeddings.
    """
    nn_path = os.path.join(output_dir, 'port2vec_nn.pkl')

    with open(nn_path, mode='rb') as file:
        nn = pickle.load(file)

    print(f'Loaded NN from: {nn_path}')

    return nn


def get_embeddings(
    word2vec: Word2Vec | KeyedVectors | str, 
    words, 
    nn: NearestNeighbors
):
    """
    Returns a matrix of embeddings for a corresponding column. If an
    entry does not have an embedding, maps to the "closest" embedding instead via
    nearest neighbors (NN). Currently, this NN approach only works if the data
    is numeric.
    """
    # Get the word2vec mapping, either directly or from a file.
    if isinstance(word2vec, Word2Vec):
        wv = word2vec.wv
    elif isinstance(word2vec, KeyedVectors):
        wv = word2vec
    else:
        wv = KeyedVectors.load(word2vec, mmap='r')

    # Just assume that the underlying words are numeric.
    vocab = set(int(key) for key in wv.key_to_index)
    vocab_list = list(vocab)

    embeddings = []

    for word in words:
        query = str(word)
        if word not in vocab:
            # Get "nearest" vocab word via nearest neighbors (assumes numeric data).
            _, indices = nn.kneighbors([[int(word)]])
            nearest_word = str(vocab_list[indices[0][0]])
            query = nearest_word
        embeddings.append(word2vec.get_vector(query, norm=False))

    result = np.stack(embeddings)

    return result


def invert_embeddings(ann: AnnoyIndex, embeddings, idx_to_vocab: dict):
    """
    Takes an array or DataFrame of embeddings for a feature and returns the original
    features as a list.
    """
    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.values

    results = []
    for embedding in embeddings:
        idx_list = ann.get_nns_by_vector(embedding, 1)
        original_word = idx_to_vocab[idx_list[0]]
        results.append(original_word)
    return results
