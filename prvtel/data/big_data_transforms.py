import dask
import dask.dataframe as dd
import pandas as pd
from dask_ml.preprocessing import StandardScaler, DummyEncoder, RobustScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
from rdt.transformers.numerical import FloatFormatter
import numpy as np
import warnings
from importlib import import_module
from random import sample
import math
import psutil


class TransformWrapper():
    """
    A wrapper class to make the preprocessing steps more modular. Put the preprocessing
    steps in a child class for a certain transform to reduce clutter from handling edge
    cases/weird interactions when working with chunked data and general boilerplate.
    """

    def __init__(self):
        self._fitted = False

    def fit(self, X, cols=None):
        """
        Fits on a Dask or Pandas DataFrame. Should set self._fitted to True.
        If cols is specified, should only fit on those columns.
        """
        raise NotImplementedError
    
    def transform(self, X):
        """
        Transforms the data. Should call check_fitted() to prevent this from
        being used before fitting. Should only transform the columns that it
        was fitted on.
        """
        raise NotImplementedError
    
    def inverse_transform(self, X):
        """
        Inverts the transformation on the data. Should call check_fitted() to prevent 
        use before fitting. Should only invert the columns that it was fitted on.
        """
        raise NotImplementedError
    
    def check_fitted(self):
        if not self._fitted:
            raise NotFittedError


class DaskStandardizer(TransformWrapper):

    def fit(self, X, cols=None):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        target_data = X[columns]

        self.standardizer = StandardScaler()
        self.standardizer.fit(target_data)
        # Save input columns (in order seen) so we can transform only these columns.
        self.input_cols = columns

        self._fitted = True

        return self
    
    def transform(self, X):
        self.check_fitted()
        results = self.standardizer.transform(X[self.input_cols])
        X[self.input_cols] = results
        return X
    
    def inverse_transform(self, X):
        self.check_fitted()
        results = self.standardizer.inverse_transform(X[self.input_cols])
        X[self.input_cols] = results
        return X


class DaskOneHotEncoder(TransformWrapper):

    def fit(self, X, cols=None):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        self.one_hot_encoder = DummyEncoder(columns)
        self.one_hot_encoder.fit(X)

        # DummyEncoder expects the exact same columns (in same order) seen during fit (including non categoricals)
        # when doing transform. The same applies to inverse_transform, where the order of output columns cannot change.
        self.input_cols = list(self.one_hot_encoder.columns_)
        self.output_cols = list(self.one_hot_encoder.transformed_columns_)

        self._fitted = True

        return self
    
    def transform(self, X):
        self.check_fitted()
        # Reorder columns so transform will work.
        results = self.one_hot_encoder.transform(X[self.input_cols])
        return results
    
    def inverse_transform(self, X):
        # TODO: This doesn't work on Dask DataFrames with unknown divisions...
        # Shouldn't be a problem during data generation, but could be an issue later.
        self.check_fitted()
        # Reorder columns so inverse_transform works.
        results = self.one_hot_encoder.inverse_transform(X[self.output_cols])
        return results


class DaskGMM(TransformWrapper):
    """Bayesian Gaussian Mixture model."""

    def fit(self, X, cols=None, sampling_frac=0.1, max_sample_size=50000):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        # Take a small sample of the data.
        X_sample = self._sample_data(
            X, 
            sampling_frac=sampling_frac, 
            max_sample_size=max_sample_size
        )

        # Fit a separate bayesian GMM on each column.
        # gmm_transforms = dict()
        scaled_col_names = []
        component_col_names = []

        # TODO: Too many tasks lead to deadlock (GIL problems?). For now, limit tasks to number of processes (as determined by Dask LocalCluster).
        from dask.distributed.deploy.utils import nprocesses_nthreads
        num_processes, _ = nprocesses_nthreads()

        # Parallelize fitting on each column.
        fit_jobs = []
        for col in columns:
            fit_job = dask.delayed(DaskGMM._fit_column)(X_sample, col)
            fit_jobs.append(fit_job)
            # Save column names.
            scaled_col_names.append(f'{col}.normalized')
            component_col_names.append(f'{col}.component')

        # Limit number of tasks done at a time.
        if isinstance(num_processes, int):
            gmm_transforms = tuple()
            for i in range(0, len(fit_jobs), num_processes):
                job_subset = fit_jobs[i:i + num_processes]
                gmm_transforms += dask.compute(*job_subset)
        else:
            gmm_transforms = dask.compute(*fit_jobs)

        self.gmm_transforms = {col: gmm for col, gmm in gmm_transforms}
        self.input_cols = columns
        self.scaled_cols = scaled_col_names
        self.component_cols = component_col_names

        self._fitted = True

        return self
    
    @classmethod
    def _fit_column(cls, X, col):
        gmm_transformer = OptimizedGMM(
            enforce_min_max_values=True,
            learn_rounding_scheme=True,
        )
        gmm_transformer.fit(X, column=col)
        return col, gmm_transformer
    
    def transform(self, X):
        """
        Applies the GMM transformation on a Dask/Pandas DataFrame. Only applies transformation
        to the columns seen during fit().
        """
        self.check_fitted()
        if isinstance(X, pd.DataFrame):
            result = self._transform_part(X)
        elif isinstance(X, dd.DataFrame):
            result = X.map_partitions(self._transform_part)
        else:
            raise TypeError('Input to transform must be a Pandas or Dask DataFrame.')
        return result
    
    def inverse_transform(self, X):
        """
        Inverts the GMM transformation on a Dask/Pandas DataFrame. Only inverts transformation
        on the columns seen during fit().
        """
        self.check_fitted()
        if isinstance(X, pd.DataFrame):
            result = self._inverse_transform_part(X)
        elif isinstance(X, dd.DataFrame):
            result = X.map_partitions(self._inverse_transform_part)
        else:
            raise TypeError('Input to inverse_transform must be a Pandas or Dask DataFrame.')
        return result
    
    def _transform_part(self, part):
        """Applies a GMM transform to an in memory partition (DataFrame)."""
        result = part
        output_cols = []
        outputs = []
        for col, gmm in self.gmm_transforms.items():
            raw_data = gmm._transform(result[col])
            outputs.append(raw_data)

            output_cols.append(f'{col}.normalized')
            output_cols.append(f'{col}.component')
        
        result = result.drop(columns=self.input_cols)
        outputs = np.concatenate(outputs, axis=1)
        result[output_cols] = outputs

        return result
    
    def _inverse_transform_part(self, part):
        """Inverts the GMM transform on an in memory partition (DataFrame)."""
        result = part
        for col, gmm in self.gmm_transforms.items():
            result = gmm.reverse_transform(result)
        return result
    
    def _sample_data(self, X, sampling_frac=0.1, max_sample_size=50000):
        """
        Creates a sample of the input data. Caps sample size to ensure
        that the data fits into memory.
        """
        # If Dask, cap the size if necessary before loading into memory.
        if isinstance(X, dd.DataFrame):
            # Get a fraction of the chunks.
            num_chunks = math.ceil(sampling_frac * X.npartitions)
            part_idx = sample(range(X.npartitions), num_chunks)
            X_sample = X.partitions[part_idx]

            # Cap sample size if necessary.
            sample_size = len(X_sample)
            if max_sample_size is not None and sample_size > max_sample_size:
                new_frac = max_sample_size / sample_size
                X_sample = X_sample.sample(frac=new_frac)

            X_sample = X_sample.compute()
        else:
            # Take a small sample of the data.
            X_sample = X.sample(frac=sampling_frac)
        return X_sample
            
    
class DaskRobustScaler(TransformWrapper):

    def fit(self, X, cols=None):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        target_data = X[columns]

        self.robust_scaler = RobustScaler()
        self.robust_scaler.fit(target_data)
        # Save input columns (in order seen) so we can transform only these columns.
        self.input_cols = columns

        self._fitted = True

        return self
    
    def transform(self, X):
        self.check_fitted()
        results = self.robust_scaler.transform(X[self.input_cols])
        X[self.input_cols] = results
        return X
    
    def inverse_transform(self, X):
        self.check_fitted()
        results = self.robust_scaler.inverse_transform(X[self.input_cols])
        X[self.input_cols] = results
        return X


class OptimizedGMM(FloatFormatter):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.

    NOTE: This code is copied from ClusterBasedNormalizer. The _transform method is modified
    to vectorize the probability predictions for efficiency.
    """

    STD_MULTIPLIER = 4
    _bgm_transformer = None
    valid_component_indicator = None

    def __init__(
        self,
        model_missing_values=None,
        learn_rounding_scheme=False,
        enforce_min_max_values=False,
        max_clusters=10,
        weight_threshold=0.005,
        missing_value_generation='random',
    ):
        # Using missing_value_replacement='mean' as the default instead of random
        # as this may lead to different outcomes in certain synthesizers
        # affecting the synthesizers directly and this is out of scope for now.
        super().__init__(
            model_missing_values=model_missing_values,
            missing_value_generation=missing_value_generation,
            missing_value_replacement='mean',
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
        )
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.output_properties = {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
        }

    def _get_current_random_seed(self):
        if self.random_states:
            return self.random_states['fit'].get_state()[1][0]
        return 0

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        sm = import_module('sklearn.mixture')

        self._bgm_transformer = sm.BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            random_state=self._get_current_random_seed(),
        )

        super()._fit(data)
        data = super()._transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm_transformer.fit(data.reshape(-1, 1))

        self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold

    def _transform(self, data):
        """Transform the numerical data.

        NOTE: Modified to vectorize transformation phase.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray.
        """
        data = super()._transform(data)
        if data.ndim > 1:
            data, model_missing_values = data[:, 0], data[:, 1]

        # Getting statistics from GMM.
        data = data.reshape((len(data), 1))
        means = self._bgm_transformer.means_.reshape((1, self.max_clusters))
        means = means[:, self.valid_component_indicator]
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape((
            1,
            self.max_clusters,
        ))
        stds = stds[:, self.valid_component_indicator]

        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (self.STD_MULTIPLIER * stds)
        component_probs = self._bgm_transformer.predict_proba(data)
        component_probs = component_probs[:, self.valid_component_indicator] + 1e-9

        # Get probabilities.
        component_probs_t = component_probs / component_probs.sum(axis=1)[:, None]
        # Higher probabilities will have higher chance of being chosen.
        scores = component_probs_t - np.random.random(component_probs_t.shape)
        selected_components = scores.argmax(axis=1)

        aranged = np.arange(len(data))
        normalized = normalized_values[aranged, selected_components].reshape([
            -1,
            1,
        ])
        normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized[:, 0]
        rows = [normalized, selected_components]
        if self.null_transformer and self.null_transformer.models_missing_values():
            rows.append(model_missing_values)

        return np.stack(rows, axis=1)  # noqa: PD013

    def _reverse_transform_helper(self, data):
        normalized = np.clip(data[:, 0], -1, 1)
        means = self._bgm_transformer.means_.reshape([-1])
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape([-1])
        selected_component = data[:, 1].round().astype(int)
        selected_component = selected_component.clip(0, self.valid_component_indicator.sum() - 1)
        std_t = stds[self.valid_component_indicator][selected_component]
        mean_t = means[self.valid_component_indicator][selected_component]
        reversed_data = normalized * self.STD_MULTIPLIER * std_t + mean_t

        return reversed_data

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.DataFrame or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        recovered_data = self._reverse_transform_helper(data)
        if self.null_transformer and self.null_transformer.models_missing_values():
            recovered_data = np.stack([recovered_data, data[:, -1]], axis=1)  # noqa: PD013

        return super()._reverse_transform(recovered_data)


class DaskLogTransformer(TransformWrapper):
    def fit(self, X, cols=None):
        """
        Fits the LogTransformWrapper to the specified columns of the data.
        
        Args:
            data: The input data (Dask or Pandas DataFrame).
            cols: List of column names to apply the log transformation.

        Returns:
            self: The fitted LogTransformWrapper instance.
        """
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)
        self.input_cols = columns
        self._fitted = True
        return self

    def transform(self, X):
        """
        Applies the log transformation to the specified columns of the data.
        
        Args:
            data: The input data (Dask or Pandas DataFrame).

        Returns:
            data: The transformed data with log applied to specified columns.
        """
        self.check_fitted()
        # copy to avoid side effects
        data = X.copy()
        for col in self.input_cols:
            # Apply log transformation with a small epsilon to avoid log(0)
            data[col] = np.log(data[col] + 1e-6)
        return data

    def inverse_transform(self, X):
        """
        Inverts the log transformation on the specified columns of the data.
        
        Args:
            data: The input data (Dask or Pandas DataFrame).

        Returns:
            data: The data with the inverse log transformation applied.
        """
        self.check_fitted()
        # copy to avoid side effects
        data = X.copy()
        for col in self.input_cols:
            data[col] = np.exp(data[col]) - 1e-6
        return data


class DaskMinMaxScaler(TransformWrapper):

    def fit(self, X, cols=None):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        target_data = X[columns]

        # Seems like MinMaxScaler doesn't respect computation graph (so will access columns that were dropped)? Fetch array directly.
        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler.fit(target_data.values)
        # Save input columns (in order seen) so we can transform only these columns.
        self.input_cols = columns

        self._fitted = True

        return self
    
    def transform(self, X):
        self.check_fitted()
        results = self.minmax_scaler.transform(X[self.input_cols])
        X[self.input_cols] = results
        return X
    
    def inverse_transform(self, X):
        self.check_fitted()
        results = self.minmax_scaler.inverse_transform(X[self.input_cols])
        X[self.input_cols] = results
        return X
    

class DaskBitEncoder(TransformWrapper):
    """
    Given integer data ranging from 0 to n, encode the bit representation 
    of the data as a one hot vector.
    """

    def fit(self, X, cols=None):
        # If cols not specified, just fit on all of them.
        columns = cols
        if columns is None:
            columns = list(X.columns)

        # TODO: Maybe determine this automatically (and different per column)?
        self.num_bits = 32

        # Save input columns (in order seen) so we can transform only these columns.
        self.input_cols = columns
        self.output_cols = [f'{col}_{i}' for col in self.input_cols for i in range(self.num_bits)]

        self._fitted = True

        return self
    
    def transform(self, X):
        self.check_fitted()
        if isinstance(X, pd.DataFrame):
            result = self._transform_part(X)
        elif isinstance(X, dd.DataFrame):
            result = X.map_partitions(self._transform_part)
        else:
            raise TypeError('Input to transform must be a Pandas or Dask DataFrame.')
        return result

    def inverse_transform(self, X):
        self.check_fitted()
        if isinstance(X, pd.DataFrame):
            result = self._inverse_transform_part(X)
        elif isinstance(X, dd.DataFrame):
            result = X.map_partitions(self._inverse_transform_part)
        else:
            raise TypeError('Input to inverse_transform must be a Pandas or Dask DataFrame.')
        return result
    
    def _transform_part(self, part):
        target_data = part[self.input_cols].astype(np.uint64).values
        target_data = np.repeat(target_data, self.num_bits, axis=1)

        mask = 2 ** np.arange(self.num_bits, dtype=np.uint64)
        mask = np.tile(mask, len(self.input_cols))

        masked_vals = target_data & mask[None, :]
        one_hot = (masked_vals != 0).astype(np.float32)

        one_hot_df = pd.DataFrame(one_hot, columns=self.output_cols)

        result = pd.concat([one_hot_df, part.drop(columns=self.input_cols)], axis=1)
        
        return result
    
    def _inverse_transform_part(self, part):
        result = part
        for col in self.input_cols:
            target_cols = [f'{col}_{i}' for i in range(self.num_bits)]
            target_data = part[target_cols].astype(np.float32)

            mask = 2 ** np.arange(self.num_bits)

            digits = mask[None, :] * target_data
            recovered_vals = digits.sum(axis=1)

            result[col] = recovered_vals
        result = result.drop(columns=self.output_cols)
        
        return result
