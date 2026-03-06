from port2vec import *
import itertools
import argparse
import os
import dask.dataframe as dd
from dask.distributed import Client


THICK_SEPARATOR = f'{'=' * 100}'
THIN_SEPARATOR = f'{'-' * 100}'


def create_embedding_data(df, wv, nn, configs):
    type_configs = configs['TYPE_CONFIGS']
    all_cols = list(df.columns)
    embedding_dfs = []
    for type, target_cols in type_configs.items():
        # Only transform columns that are actually in the data (preserve order).
        valid_cols = [col for col in all_cols if col in set(target_cols)]
        for col in valid_cols:
            # Ports and protocols are assumed to be integers.
            embeddings = get_embeddings(wv, df[col].astype(int), nn)
            embedding_size = embeddings.shape[1]
            col_names = [f'{col}_{i}' for i in range(embedding_size)]
            result = pd.DataFrame(embeddings, columns=col_names)
            embedding_dfs.append(result)
    result_df = pd.concat([df] + embedding_dfs, axis=1)

    # Remove original column labels
    original_cols = itertools.chain.from_iterable([target_cols for _, target_cols in type_configs.items()])
    result_df = result_df.drop(columns=original_cols)
    return result_df


def invert_embedding_data(df, wv, configs):
    embedding_names = []
    type_configs = configs['TYPE_CONFIGS']

    # NOTE: Ugly, but have to do this since ANN cannot be hashed. This should just load from
    # disk since it was already built from a prior call.
    ann_map = create_ann_map(wv, df, configs)

    original_cols = set(df.columns)

    for type, target_cols in type_configs.items():

        # Get ANN and vector indices map for data type.
        type_ann, type_idx_map = ann_map[type]

        for col in target_cols:
            embedding_size = configs['WORD2VEC_CONFIGS']['embedding_size']
            col_names = [f'{col}_{i}' for i in range(embedding_size)]

            # Check that all embedding dimensions are present.
            # If not, model embedding size bigger than what the data has (need different model).
            if not set(col_names).issubset(original_cols):
                continue

            embeddings = df[col_names]

            inverted_data = invert_embeddings(type_ann, embeddings.astype(int), type_idx_map)

            df[col] = inverted_data

            embedding_names += col_names

    # Remove labels of embedding features.
    df = df.drop(columns=embedding_names)

    return df


def create_ann_map(wv, df, configs, rebuild=False):
    # Build an index for each type of column label separately.
    # NOTE: I guess this is so that port embeddings don't clash with protocol embeddings? 
    type_configs = configs['TYPE_CONFIGS']
    save_dir = configs['GENERAL']['save_dir']
    w2v_configs = configs['WORD2VEC_CONFIGS']

    ann_map = {}
    for type, target_cols in type_configs.items():
        # Try to load existing data if possible (unless Word2Vec retrained).
        ann, idx_to_vocab = build_ann_index(
            df,
            word2vec=wv,
            target_cols=target_cols,
            col_type=type,
            embedding_size=w2v_configs['embedding_size'],
            ann_n_trees=w2v_configs['ann_n_trees'],
            output_dir=save_dir,
            rebuild=rebuild
        )
        ann_map[type] = (ann, idx_to_vocab)

    return ann_map


def create_nn(wv, configs, rebuild=False):
    # Try to load existing data if possible (unless Word2Vec retrained).
    save_dir = configs['GENERAL']['save_dir']

    nn = build_nn(wv, save_dir, rebuild=rebuild)

    return nn


def get_command_line_args():
    # Initialize
    parser = argparse.ArgumentParser(description='Either training or applying Word2Vec embeddings on port and protocol data.')

    # File related (inputs, saving output).
    parser.add_argument('--input_data', nargs='+', required=True, help='Path to CSV file. Data to apply transformation on.')
    parser.add_argument('--train_data', default=None, help='Data to train Word2Vec on. Ignored if --train_model not passed.')
    parser.add_argument('--model_dir', default='port2vec', help='Directory to save Word2Vec, ANN index, etc.')
    parser.add_argument('--result_path', default=None, help='Where to save transformed (or inverse transformed) data.')
    parser.add_argument('--blocksize', default='default', help='How to chunk data to be transformed.')
    parser.add_argument('--single_file', default=False, action='store_true', help='Whether to save the output as a single file.')


    # Model configuration related.

    # NOTE: Make sure that you specify all parameters correctly (embedding size, trees), otherwise, it cannot load the word2vec or ANN correctly.
    parser.add_argument('--mode', default='transform', choices=['transform', 'invert'], help='Either "transform" or "invert". Corresponds to training, '
                        'converting ports/protocols to embeddings, and reversing that process respectively.')
    parser.add_argument('--embedding_size', type=int, default=10, help='Size of Word2Vec embedding. You must enter the correct embedding size, even when loading the model.')
    parser.add_argument('--ann_trees', type=int, default=100, help='Number of trees to generate when making ANN index. Not needed unless rebuilding ANN index ("train" mode).')

    # Training specific.
    parser.add_argument('--force_retrain', default=False, action='store_true', help='Forcibly (re)train Word2Vec on the input to --train_data.')
    parser.add_argument('--port_columns', type=str, nargs='+', default=['L4_SRC_PORT', 'L4_DST_PORT'],
                          help='Space-separated list of column labels that are considered network port numbers.'
                          'Example: --port_columns L4_SRC_PORT L4_DST_PORT')
    parser.add_argument('--protocol_columns', type=str, nargs='+', default=['PROTOCOL'],
                          help='Space-separated list of labels that are considered network protocol numbers.'
                          'Example: --protocol_columns PROTOCOL')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_command_line_args()

    # Ensure creation of necessary directories.
    os.makedirs(args.model_dir, exist_ok=True)

    folder = os.path.dirname(os.path.abspath(args.result_path))
    os.makedirs(folder, exist_ok=True)

    # Display configs (command line args).
    print(f'{THICK_SEPARATOR}\nCONFIGURATION\n{THIN_SEPARATOR}')
    for key, value in vars(args).items():
        print(f"{key.upper()}: {value}")
    print(THICK_SEPARATOR)


    # Initialize configs for training.
    CONFIGS = {
        'GENERAL': {
            'save_dir': args.model_dir
        },
        'WORD2VEC_CONFIGS': {
            'embedding_size': args.embedding_size,
            'ann_n_trees': args.ann_trees,
        },
        'TYPE_CONFIGS': {
            'PORT': args.port_columns,
            'PROTOCOL': args.protocol_columns
        }
    }

    blocksize = None if args.blocksize == 'None' else args.blocksize
    
    # Normalize numeric columns to float to avoid metadata mismatch issues.
    # TODO: Is there a better way to do this?
    infer_types = dd.read_csv(args.input_data, blocksize=blocksize)
    col_types = infer_types.dtypes.to_dict()
    for col, dtype in col_types.items():
        if dtype == np.int64:
            col_types[col] = 'float64'
    ddf = dd.read_csv(args.input_data, blocksize=blocksize, dtype=col_types)

    # Must rebuild everything if Word2Vec retrained.
    force_rebuild = args.force_retrain

    # Get training data for Word2Vec. This is silently ignored if we are preloading a model.
    train_path = args.train_data
    if isinstance(train_path, str) and os.path.exists(train_path):
        train_df = pd.read_csv(args.train_data)
    else:
        print('WARNING: Invalid training data provided. This could cause an error if the Word2Vec has to be (re)trained.')
        train_df = None

    print(f'TRAINING/LOADING PORT2VEC\n{THIN_SEPARATOR}')

    # Train Word2Vec on every column to be encoded.
    sentence_columns = list(itertools.chain.from_iterable([target_cols for _, target_cols in CONFIGS['TYPE_CONFIGS'].items()]))
    wv, rebuilt = build_word2vec(
        train_df,
        word2vec_cols=sentence_columns,
        force_rebuild=force_rebuild,
        output_dir=args.model_dir
    )

    # NOTE: If Word2Vec is retrained, everything else will be rebuilt as well.

    # Dictionary mapping port type to a tuple of the form (ann, idx_to_vocab).
    ann_map = create_ann_map(wv, train_df, CONFIGS, rebuild=rebuilt)

    # Nearest neighbor model for generating embeddings.
    nn = create_nn(wv, configs=CONFIGS, rebuild=rebuilt)

    print(THICK_SEPARATOR)

    client = Client()


    # Transform data w/ embedding model.
    mode = args.mode
    if mode == 'transform':
        # Converting ports and protocols to embeddings.
        print(f'CREATING EMBEDDINGS\n{THIN_SEPARATOR}')
        print(f'Dashboard Link: {client.dashboard_link}')
        
        ddf = ddf.map_partitions(create_embedding_data, wv=wv, nn=nn, configs=CONFIGS)
    elif mode == 'invert':
        # Converting embedding columns (of the form COL_0, COL_1, ..., COL_N) back.
        print(f'REVERSING EMBEDDINGS\n{THIN_SEPARATOR}')
        print(f'Dashboard Link: {client.dashboard_link}')

        ddf = ddf.map_partitions(invert_embedding_data, wv=wv, configs=CONFIGS)

    print(f'Writing results to: {args.result_path}...')

    directory = os.path.dirname(args.result_path)
    dataset_name, file_type = os.path.splitext(os.path.basename(args.result_path))
    if file_type == '.csv':
        ddf.to_csv(args.result_path, single_file=args.single_file, index=False)
    elif file_type == '.parquet':
        leading_zeros = len(str(ddf.npartitions))
        name_gen = lambda i: dataset_name.replace('*', f'{i:0{leading_zeros}d}') + '.parquet' if '*' in dataset_name else dataset_name + f'_{i:0{leading_zeros}d}' + '.parquet'
        # TODO: Not sure about performance of this...
        if args.single_file:
            ddf = ddf.repartition(npartitions=1)
        ddf.to_parquet(directory, name_function=name_gen, write_index=False)


    