from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.lib.utils import display_bayesian_network
from argparse import ArgumentParser
import pandas as pd
import time
import os

def preprocess_data(input_data_filepath, processed_data_path, ip_cols=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], drop_cols=None, categorical_threshold=10):
    # load data to pandas dataframe
    input_df = pd.read_csv(input_data_filepath)

    # only take the following columns for the input dataset: IPV4_SRC_ADDR,L4_SRC_PORT,IPV4_DST_ADDR,L4_DST_PORT,PROTOCOL,IN_BYTES,IN_PKTS,OUT_BYTES,OUT_PKTS
    if drop_cols is not None:
        input_df = input_df.drop(columns=drop_cols)
    # Convert IP addresses to integer with bitwise encoding.
    for col in ip_cols:
        input_df[col] = bitwise_encoding_ip_col(input_df[col])  

    input_df.to_csv(processed_data_path, index=False)
    # Find all column/features with categorical value

    original_categorical_columns = []
    categorical_len_count = 0
    for col in input_df:
        # Do not process the value
        if len(input_df[col].unique()) <= categorical_threshold:
            original_categorical_columns.append(col)
            categorical_len_count += len(input_df[col].unique())

    return input_df, original_categorical_columns


def bitwise_encoding_ip_col(col, reverse=False):
    if reverse:
        return col.apply(lambda x: '.'.join(str(x // 256**i % 256) for i in range(3, -1, -1)))
    return col.apply(lambda x: int(x.split('.')[0]) * 256**3 + int(x.split('.')[1]) * 256**2 + int(x.split('.')[2]) * 256 + int(x.split('.')[3]))


def apply_privbayes_noise(args):
    input_data = args.input_data
    processed_data_path = args.processed_data_path
    
    # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
    # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
    epsilon = args.epsilon

    # location of two output files
    file_prefix = os.path.splitext(input_data)[0]

    description_file = f'{file_prefix}_description_epsilon{epsilon}.json'
    synthetic_data_path = f'{file_prefix}_dp_noise_epsilon{epsilon}.csv' if args.synthetic_data_path is None else args.synthetic_data_path

    # An attribute is categorical if its domain size is less than this threshold.
    # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
    threshold_value = 10

    input_df, original_categorical_columns = preprocess_data(
        input_data_filepath=input_data, processed_data_path=processed_data_path, 
        drop_cols=args.drop_cols, ip_cols=args.ip_cols, categorical_threshold=threshold_value)

    dict_categorical_columns = {item: True for item in original_categorical_columns}

    # specify categorical attributes
    categorical_attributes = dict_categorical_columns

    # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
    degree_of_bayesian_network = 1

    # Number of tuples generated in synthetic dataset: same as original
    num_tuples_to_generate = input_df.shape[0]
    start = time.time()

    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=processed_data_path,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes)
    describer.save_dataset_description_to_file(description_file)
    display_bayesian_network(describer.bayesian_network)

    generator = DataGenerator()
    print("before generator: ", time.time()-start)
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    syn_df = generator.synthetic_dataset

    # reverse the bitwise encoding
    for col in args.ip_cols:
        syn_df[col] = bitwise_encoding_ip_col(syn_df[col], reverse=True)

    syn_df.to_csv(synthetic_data_path, index=False)
    print("after generator: ", time.time()-start)

    print("-------Noise-added and Data saved---------")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--drop_cols', nargs='+', default=None, help='Comma-separated list of columns to drop')
    parser.add_argument('--ip_cols', nargs='+', default=None, help='Comma-separated list of IP columns to reverse bitwise encoding')
    parser.add_argument('--processed_data_path', type=str, default='processed_data.csv', help='Path to save the preprocessed dataset')
    parser.add_argument('--epsilon', type=float, default=2.0, help='Epsilon value for differential privacy')
    parser.add_argument('--synthetic_data_path', type=str, default=None, help='Path to save the synthetic dataset')
    args = parser.parse_args()

    apply_privbayes_noise(args=args)
