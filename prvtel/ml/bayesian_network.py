from DataSynthesizer.DataDescriber import DataDescriber

def build_bayesian_network(input_data_path, original_categorical_columns, degree_of_bayesian_network=2, epsilon=2):
        threshold_value = 10

        dict_categorical_columns = {item: True for item in original_categorical_columns}
        categorical_attributes = dict_categorical_columns

        describer = DataDescriber(category_threshold=threshold_value)
        bn_structure = describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data_path, 
                                                                epsilon=epsilon, 
                                                                k=degree_of_bayesian_network,
                                                                attribute_to_is_categorical=categorical_attributes)

        return bn_structure