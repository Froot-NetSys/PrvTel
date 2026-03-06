import dask.dataframe as dd
from dask.distributed import get_client
import pandas as pd
import os
import math
import numpy as np
from enum import Enum
from typing import List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import itertools

# User code.
from .big_data_transforms import DaskGMM, DaskOneHotEncoder, DaskStandardizer, DaskRobustScaler, DaskLogTransformer, DaskMinMaxScaler, DaskBitEncoder
from ..utils import get_file_size

class PreprocessingMethod(Enum):
    LOG = 'log'
    GMM_LOG = 'GMM_log'
    GMM = 'GMM'
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    LOG_ROBUST = 'log_robust'
    LOG_MINMAX = 'log_minmax'
    LOG_MINMAX_GMM = 'log_minmax_gmm'
    BIT = 'bit'

@dataclass
class PreprocessingResult:
    data: dd.DataFrame
    transforms: List[Any]
    num_continuous: int
    num_categories_per_col: List[int]

class BasePreprocessor:
    @staticmethod
    def _one_hot_encode_categoricals(
        data: Union[dd.DataFrame, pd.DataFrame],
        cat_cols: List[str]
    ) -> Tuple[dd.DataFrame, DaskOneHotEncoder, List[int]]:
        """Helper function to handle one-hot encoding of categorical features."""
        ddf = data
        if isinstance(data, dd.DataFrame):
            ddf = ddf.categorize(cat_cols)
        elif isinstance(data, pd.DataFrame):
            ddf[cat_cols] = ddf[cat_cols].astype('category')

        num_cats_per_col = [len(ddf[col].cat.categories) for col in cat_cols]
        num_cats = sum(num_cats_per_col)
        
        onehot = DaskOneHotEncoder()
        onehot.fit(ddf, cols=cat_cols)
        ddf = onehot.transform(ddf)

        reordered_cols = list(ddf.columns[-num_cats:]) + list(ddf.columns[:-num_cats])
        ddf = ddf.loc[:, reordered_cols]

        return ddf, onehot, num_cats_per_col

class ContinuousPreprocessor(BasePreprocessor):
    @staticmethod
    def log_transform(data: dd.DataFrame, cont_cols: List[str]) -> Tuple[dd.DataFrame, List[Any]]:
        """Apply log transform to continuous columns."""
        transforms = []
        log_transformer = DaskLogTransformer()
        log_transformer.fit(data, cols=cont_cols)
        transformed_data = log_transformer.transform(data)
        transforms.append(log_transformer)
        return transformed_data, transforms

    @staticmethod
    def gmm_transform(
        data: dd.DataFrame,
        cont_cols: List[str],
        sampling_frac: float = 0.1,
        max_sample_size: int = 50000
    ) -> Tuple[dd.DataFrame, List[Any], List[str]]:
        """Apply GMM transformation to continuous columns."""
        gmm = DaskGMM()
        gmm.fit(data, cols=cont_cols, sampling_frac=sampling_frac, max_sample_size=max_sample_size)
        transformed_data = gmm.transform(data)
        return transformed_data, [gmm], gmm.component_cols

class PreprocessingFactory:
    def __init__(self):
        self.continuous_processor = ContinuousPreprocessor()
        self.methods = {
            PreprocessingMethod.LOG.value: self._log_preprocessing,
            PreprocessingMethod.GMM_LOG.value: self._gmm_with_log_preprocessing,
            PreprocessingMethod.GMM.value: self._gmm_preprocessing,
            PreprocessingMethod.STANDARD.value: self._standard_preprocessing,
            PreprocessingMethod.MINMAX.value: self._minmax_preprocessing,
            PreprocessingMethod.ROBUST.value: self._robust_preprocessing,
            PreprocessingMethod.LOG_ROBUST.value: self._log_robust_preprocessing,
            PreprocessingMethod.LOG_MINMAX.value: self._log_minmax_preprocessing,
            PreprocessingMethod.LOG_MINMAX_GMM.value: self._log_minmax_gmm_preprocessing,
            PreprocessingMethod.BIT.value: self._encode_bits,
        }

    def process_categoricals(
        self,
        data: dd.DataFrame,
        cat_cols: List[str],
        transforms: Optional[List[Any]] = None
    ) -> PreprocessingResult:
        """
        Applies a one hot transformation on corresponding categorical columns and returns
        associated metadata between continuous/categorical features. Should be called on
        entire dataset after other transforms are applied.

        Args:
            data: The data with other preprocessing already applied.
            cont_cols: The final list of continuous features in the dataset.
            cat_cols: The final list of categorical features (after other transformations applied).
            transforms: The list of transforms to add the one hot encoder to.
        Returns:
            See preprocess_large_data.
        """
        transforms = transforms or []
        ddf = data
        num_cats_per_col = []

        if cat_cols:
            print(f'One hot encoding {cat_cols}...')
            ddf, onehot, num_cats_per_col = self.continuous_processor._one_hot_encode_categoricals(
                data=ddf,
                cat_cols=cat_cols
            )
            transforms.append(onehot)

        num_conts = len(ddf.columns) - sum(num_cats_per_col)

        return PreprocessingResult(ddf, transforms, num_conts, num_cats_per_col)

    def _log_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Applies log transform to continuous columns."""
        print(f'Applying log transform to {target_cols}...')
        ddf, transforms = self.continuous_processor.log_transform(data, target_cols)
        return ddf, [], transforms

    def _gmm_with_log_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        sampling_frac: float = 0.1,
        max_sample_size: int = 50000,
        **kwargs
    ):
        """Applies Bayesian GMM to continuous columns after applying a log transform."""
        transforms = []

        print(f'Applying log GMM transform to {target_cols}...')
        
        # Log transform
        ddf, log_transforms = self.continuous_processor.log_transform(data, target_cols)
        transforms.extend(log_transforms)

        # GMM transform
        ddf, gmm_transforms, component_cols = self.continuous_processor.gmm_transform(
            ddf, target_cols, sampling_frac, max_sample_size
        )
        transforms.extend(gmm_transforms)

        # Return the new GMM cols for one hot encoding.
        return ddf, component_cols, transforms

    def _gmm_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        sampling_frac: float = 0.1,
        max_sample_size: int = 50000,
        **kwargs
    ):
        """Applies Bayesian GMM to continuous columns."""
        print(f'Applying GMM to {target_cols}...')
        ddf, transforms, component_cols = self.continuous_processor.gmm_transform(
            data, target_cols, sampling_frac, max_sample_size
        )
        # Return the new GMM cols for one hot encoding.
        return ddf, component_cols, transforms

    def _standard_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Applies standardization to continuous columns."""
        transforms = []
        ddf = data

        print(f'Applying standardization to {target_cols}...')
        standardizer = DaskStandardizer()
        standardizer.fit(ddf, cols=target_cols)
        ddf = standardizer.transform(ddf)
        transforms.append(standardizer)

        return ddf, [], transforms
    
    def _minmax_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Scales continuous columns to [0, 1] range."""
        transforms = []
        ddf = data

        print(f'Applying Min-Max scaling to {target_cols}...')
        minmax_scaler = DaskMinMaxScaler()
        minmax_scaler.fit(ddf, cols=target_cols)
        ddf = minmax_scaler.transform(ddf)
        transforms.append(minmax_scaler)

        return ddf, [], transforms

    def _robust_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Scales continuous columns based on median and IQR."""
        transforms = []
        ddf = data

        print(f'Applying robust scaling to {target_cols}...')
        robust_scaler = DaskRobustScaler()
        robust_scaler.fit(ddf, cols=target_cols)
        ddf = robust_scaler.transform(ddf)
        transforms.append(robust_scaler)

        return ddf, [], transforms

    def _log_robust_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        epsilon: float = 1e-8,
        **kwargs
    ):
        """Applies log transform followed by robust scaling."""
        transforms = []
        
        # Log transform
        print(f'Applying log transform to {target_cols}...')
        ddf, log_transforms = self.continuous_processor.log_transform(data, target_cols)
        transforms.extend(log_transforms)

        # Robust scaling
        print(f'Applying robust scaling to {target_cols}...')
        robust_scaler = DaskRobustScaler()
        robust_scaler.fit(ddf, cols=target_cols)
        ddf = robust_scaler.transform(ddf)
        transforms.append(robust_scaler)

        return ddf, [], transforms

    def _log_minmax_gmm_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        sampling_frac: float = 0.1,
        max_sample_size: int = 50000,
        **kwargs
    ):
        """Applies log transform, Min-Max scaling, and then GMM to continuous columns."""
        transforms = []

        # Log transform
        print(f'Applying log transform to {target_cols}...')
        ddf, log_transforms = self.continuous_processor.log_transform(data, target_cols)
        transforms.extend(log_transforms)

        # Min-Max scaling
        print(f'Applying Min-Max scaling to {target_cols}...')
        minmax_scaler = DaskMinMaxScaler()
        minmax_scaler.fit(ddf, cols=target_cols)
        ddf = minmax_scaler.transform(ddf)
        transforms.append(minmax_scaler)

        # GMM transform
        print(f'Applying GMM to {target_cols}...')
        ddf, gmm_transforms, component_cols = self.continuous_processor.gmm_transform(
            ddf, target_cols, sampling_frac, max_sample_size
        )
        transforms.extend(gmm_transforms)

        # Return the new GMM cols for one hot encoding.
        return ddf, component_cols, transforms

    def _log_minmax_preprocessing(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Applies log transform, then Min-Max scaling."""
        transforms = []

        # Log transform
        print(f'Applying log transform to {target_cols}...')
        ddf, log_transforms = self.continuous_processor.log_transform(data, target_cols)
        transforms.extend(log_transforms)

        # Min-Max scaling
        print(f'Applying Min-Max scaling to {target_cols}...')
        minmax_scaler = DaskMinMaxScaler()
        minmax_scaler.fit(ddf, cols=target_cols)
        ddf = minmax_scaler.transform(ddf)
        transforms.append(minmax_scaler)

        return ddf, [], transforms

    def _encode_bits(
        self,
        data: dd.DataFrame,
        target_cols: List[str],
        **kwargs
    ):
        """Applies Bayesian GMM to continuous columns after applying a log transform."""
        transforms = []
        ddf = data

        # Encode original categorical columns to array of 1s and 0s.
        print(f'Creating bit array representation of {target_cols}...')
        bit_encoder = DaskBitEncoder()
        bit_encoder.fit(ddf, cols=target_cols)
        ddf = bit_encoder.transform(ddf)
        transforms.append(bit_encoder)

        # Return the bit columns to be one hot encoded. This is for compatability with rest of code...
        return ddf, bit_encoder.output_cols, transforms


def read_large_data(
        input_file_paths, 
        excluded_cols=['time', 'timestamp'], 
        output_dtype='float32',
        file_format='csv',
        categoricals='auto',
        blocksize='auto',
        **kwargs
):
    """
    Reads one or more csv/parquet files into a Dask DataFrame. Distinguishes between
    continuous and categorical features.

    Parameters
    ----------
    input_file_paths : list[str] or str
        A list of file paths or a glob string of the files.
    
    excluded_cols : list[str] or str, default=['time', 'timestamp']
        The column labels to exclude. Can either be a list or a string, with labels
        separated by whitespace. If an empty string/list is passed, nothing is dropped.
        Will only drop the columns that are actually present in the data.

    dtype : str or dtype, default='float32'
        Data type to cast input data to. If None, no casting will be done.

    file_format : {'csv', 'parquet'}, default='csv'
        The expected file type of the input.

    categoricals : list[str] or str or 'auto'
        If 'auto', make a pass over the data to automatically determine categorical columns
        based on number of unique values. Otherwise you can specify the categorical columns
        manually. This can be done either with a whitespace separated string or a list. If
        no columns are categorical, an empty string or list should be given.

    blocksize: int, str, 'auto', or None, default='auto'
        The size of the chunks (blocks) in bytes that the input files should be divided into.
        If 'auto', automatically splits data across workers. If None, uses one chunk per file.

    Returns
    -------
    ddf_and_col_names : tuple[dask.dataframe.DataFrame, list[str], list[str]]
        Returns a tuple containing the data (Dask DataFrame), the
        names of the continuous columns (list), and the names of the
        categorical columns.
    """
    
    if file_format == 'csv':
        reader = dd.read_csv
    elif file_format == 'parquet':
        reader = dd.read_parquet
    else:
        raise ValueError(f'{file_format} not supported. Can only read csv or parquet')
    
    # Automatically chunk input so that it can be distributed across multiple CPUs if possible.
    if blocksize == 'auto':
        blocksize = determine_blocksize(input_file_paths=input_file_paths)
    elif blocksize == 'None':
        blocksize = None
        
    # Getting the ddf (Dask Dataframe) to compute statistics. Cast to dtype if needed.
    ddf = reader(input_file_paths, blocksize=blocksize, **kwargs)
    if output_dtype is not None:
        ddf = ddf.astype(output_dtype)

    # Drop any unwanted columns.
    if isinstance(excluded_cols, str):
        excluded_cols = [col for col in excluded_cols.split()]
    if excluded_cols:
        dropped_cols = set(ddf.columns) & set(excluded_cols)
        ddf = ddf.drop(columns=dropped_cols)

    original_continuous_columns = []
    original_categorical_columns = []
    # Find categorical columns automatically. If not, assume
    # all columns are continuous.
    columns = list(ddf.columns)
    if categoricals == 'auto':
        for col in columns:
            # Do not process the value
            n_uniques = ddf[col].unique().compute()
            num_cats = len(n_uniques)
            if num_cats <= 10:
                original_categorical_columns.append(col)
            else:
                original_continuous_columns.append(col)
    elif isinstance(categoricals, str):
        original_categorical_columns = [col for col in categoricals.split() if col in columns]
        original_continuous_columns = [col for col in columns if col not in original_categorical_columns]
    elif isinstance(categoricals, list):
        original_categorical_columns = [col for col in categoricals if col in columns]
        original_continuous_columns = [col for col in columns if col not in categoricals]
    else:
        raise ValueError(
            'Value passed to "categoricals" must be either "auto", a string of column labels separated by white space, or a list.'
        )

    return ddf, original_continuous_columns, original_categorical_columns


def preprocess_large_data(
        data, 
        cont_cols,
        cat_cols,
        pre_proc_method='GMM',
        pre_proc_config=None,
        output_dtype='float32', 
        *args, 
        **kwargs
    ):
    """
    Dispatch function to apply specified preprocessing method to data.

    Parameters
    ----------
    data : dd.dataframe.DataFrame
        The data to be preprocessed.
    cont_cols : list[str]
        The names of the continuous features of the data.
    cat_cols : list[str]
        The names of the categorical features of the data.
    pre_proc_method : str, default='GMM'
        Specifies the preprocessing method to be used. Must be one of the values in PreprocessingMethod enum.
    pre_proc_config : dict, default=None
        Specifies the preprocessing methods to be used on select columns. A dictionary with up to two keys: 
        'custom' and 'default', which contain information on the method used and columns to transform. Prioritized
        over pre_proc_method when both are specified.
    output_dtype : str or dtype or None
        The data type to cast the output to. If None, no casting will be done.
    *args, **kwargs
        Additional arguments passed to the preprocessing method.
    
    Returns
    -------
    data_transforms_and_col_stats : tuple[dask.dataframe.DataFrame, list[TransformWrapper], int, list[int]]
        Returns a tuple containing the transformed data (dask.DataFrame)
        and a list of the transforms used. Also returns the number of continuous features
        and a list of the number of categories for each categorical feature. 
        
        The final DataFrame should have all the categorical features at the beginning. Each
        transform is an instance of TransformWrapper, which implements fit, transform, and inverse_transform.
    """
    method_configs = []
    # Prioritize config file over pre_proc_method.
    if pre_proc_config is not None:
        method_configs = pre_proc_config.get('custom', [])
        default_config = pre_proc_config.get('default', [])
        if default_config:
            original_cols = list(data.columns)
            visited_cols = set(itertools.chain.from_iterable(method_config['columns'] for method_config in method_configs))
            # Categoricals should also be excluded (treated differently).
            visited_cols = visited_cols | set(cat_cols)
            other_cols = [col for col in original_cols if col not in visited_cols]
            # Transform remaining columns with default method (if specified).
            for config in default_config:
                config['columns'] = other_cols

            method_configs.extend(default_config)
    elif pre_proc_method is not None:
        # Old behavior: apply transform on continuous columns. One hot cat_cols after.
        method_configs = [
            {
                'method': pre_proc_method,
                'columns': cont_cols
            }
        ]

    # Apply transforms if there are any transforms specified.
    if method_configs:
        factory = PreprocessingFactory()

        ddf = data
        transforms = []
        for method_config in method_configs:
            method = method_config['method']
            target_cols = method_config['columns']

            if not isinstance(method, PreprocessingMethod):
                try:
                    method = PreprocessingMethod(method)
                except ValueError:
                    raise ValueError(
                        f'Invalid preprocessing method: {method}. '
                        f'Must be one of {[m.value for m in PreprocessingMethod]}'
                    )
                
                ddf, new_cat_cols, new_transforms = factory.methods[method.value](
                    data=ddf,
                    target_cols=target_cols,
                    **kwargs
                )
                # Collect all extra categorical columns created during preprocessing. One hot all at once.
                cat_cols += new_cat_cols
                transforms += new_transforms

        # One hot encode the categoricals and collect resulting metadata as the final step.
        result = factory.process_categoricals(
            data=ddf,
            cat_cols=cat_cols,
            transforms=transforms
        )
    else:
        result = PreprocessingResult(data, [], len(cont_cols), [])
        

    if output_dtype is not None:
        result.data = result.data.astype(output_dtype)
    
    return result.data, result.transforms, result.num_continuous, result.num_categories_per_col


def determine_blocksize(input_file_paths, max_block_size=int(1e+9)):
    """
    Finds a blocksize (in bytes) that will split the input evenly across
    available CPU cores. If no Client available, just use defaults.
    """
    try:
        client = get_client()
        input_size = get_file_size(input_file_paths)
        # Seems like splitting data across threads is fine for parallelism (at least for reading data).
        # Not sure if this would affect more complex operations though.
        num_cores = sum(client.ncores().values())
        bytes_per_core = math.ceil(input_size / num_cores)
        blocksize = min(max_block_size, bytes_per_core)
    except:
        # Use default blocksize
        blocksize = 'default'
    return blocksize