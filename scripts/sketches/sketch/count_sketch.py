import pandas as pd
import numpy as np
import math
import numpy as np


def cs_part(df, cols, rows, rho):
    """
    Create count sketches for each column of a DataFrame.
    Packages output as a dictionary, where each key is a column label
    pointing to a corresponding count sketch.
    """
    cs_dict = {}
    for col in df.columns:
        cs_dict[col] = FasterCountSketch(cols, rows, rho=rho)
    for col in df.columns:
        cs = cs_dict[col]
        cs.batch_update(df[col])
    return cs_dict


def create_count_sketches(ddf, cols=2000, rows=10, rho=2):
    """Creates a dictionary of count sketches, one for each column, on a Dask DataFrame."""
    columns = list(ddf.columns)

    # Each partition returns a dict, with a CountSketch for each column.
    counts = ddf.map_partitions(cs_part, cols=cols, rows=rows, rho=rho).compute()
    cs_dicts = list(counts.values)

    # Aggregate the sketches across partitions.
    col2CS = {}
    for col in columns:
        cs_list = [cs_dict[col] for cs_dict in cs_dicts]
        col2CS[col] = FasterCountSketch.combine(cs_list)
    return col2CS


def dcs_part(df, universe, gamma, rho):
    """
    Create DCS for each column of a DataFrame.
    Each key in the output dictionary is a column label
    pointing to a corresponding DCS.
    """
    dcs_dict = {}
    for col in df.columns:
        dcs_dict[col] = DCS(universe, gamma, rho=rho)
    for col in df.columns:
        dcs = dcs_dict[col]
        dcs.batch_update(df[col])
    return dcs_dict


def create_dcs(ddf, universe=2**30, gamma=0.0325, rho=2):
    """Creates a dictionary of count sketches, one for each column, on a Dask DataFrame."""
    columns = list(ddf.columns)

    # Each partition returns a dict, with a CountSketch for each column.
    counts = ddf.map_partitions(
        dcs_part, 
        universe=universe, 
        gamma=gamma, 
        rho=rho
    ).compute()
    dcs_dicts = list(counts.values)

    # Aggregate the sketches across partitions.
    col2DCS = {}
    for col in columns:
        dcs_list = [dcs_dict[col] for dcs_dict in dcs_dicts]
        col2DCS[col] = DCS.combine(dcs_list)
    return col2DCS


class FasterCountSketch():
    """Worked faster in tests on Cisco..."""

    def __init__(self, t, d, rho = None):
        self.t = t # columns
        self.d = d # rows
        self.gamma = 1.0 / t
        self.beta = 1.0 / math.exp(self.d)
        # Keep this around to make updating hash matrix easier
        self.row_indices = np.arange(self.d, dtype=int)

        self.C = []
        self.sigma = None
        if not rho:
            self.C = np.zeros((self.d, self.t))
        else:
            self.sigma = math.sqrt(math.log(2.0 / self.beta) / rho)
            self.C = np.random.normal(0, self.sigma, (self.d, self.t))
    
    def _hash_col(self, col, i=0):
        """
        Hash entire column at once for a given row in the table.
        Returns results as numpy arrays.
        """
        # Get column assignments.
        data = col.astype('string') + f'_{i}'
        bucket_hashes = pd.util.hash_pandas_object(data, index=False)
        buckets = bucket_hashes % self.t
        # Get updates.
        value_hashes = pd.util.hash_pandas_object(data + f'_{i}', index=False)
        values = (value_hashes % 2).astype(int)
        values[values == 0] = -1
        return buckets, values

    def batch_update(self, col, weight=1):
        """Update row by row for an entire column (pd.Series)."""
        for i in range(self.d):
            buckets, values = self._hash_col(col, i)
            temp = pd.Series(values, index=buckets)
            updates = temp.groupby(temp.index).sum()

            indices = updates.index.values
            values = updates.values

            self.C[i, indices] += values
        return self
    
    def batch_query(self, items):
        """
        Query an entire pd.Series of unique values.
        Returns a numpy array of the estimated frequencies.
        """
        all_values = []
        for i in range(self.d):
            buckets, vals = self._hash_col(items, i)
            entries = self.C[i, buckets] * vals
            all_values.append(entries)

        freq_matrix = np.stack(all_values)
        freqs = np.median(freq_matrix, axis=0)
        return freqs
    
    @classmethod
    def combine(cls, cs_list):
        # Aggregate in first one.
        first = cs_list[0]
        for cs in cs_list[1:]:
            first.C += cs.C
        return first


class DCS():
    def __init__(self, universe, gamma, rho = None):
        # assume the universe is a power of 2
        #w = 1/gamma sqrt(logU log(logU/gamma)) columns
        #d = log(logU /gamma) rows
        self.totalsize = 0
        self.U = universe
        self.gamma = gamma

        self.columns = math.ceil( (1.0/self.gamma) * math.sqrt(math.log(self.U)*math.log(math.log(self.U)/self.gamma)) )
        # self.rows = math.ceil(math.log(math.log(self.U)/self.gamma))
        self.rows = 2 # Hack for large data.

        self.total_levels = math.ceil ( math.log2(universe) )
        self.subdomains = []
        for i in range(self.total_levels):
            if rho:
                self.subdomains.append( FasterCountSketch(self.columns, self.rows, 1.0 * rho / self.total_levels) )
            else:
                self.subdomains.append( FasterCountSketch(self.columns, self.rows) )
            self.totalsize += self.rows*self.columns

    def batch_update(self, col, weight = 1):
        """Update all subdomains with an entire column."""
        for j in range(0, self.total_levels):
            self.subdomains[j].batch_update(col, weight)

            col = col // 2

    def rank(self, x):
        result = 0
        for i in range(self.total_levels):
            if x % 2 == 1:
                result += self.subdomains[i].query(x - 1)
            x = x // 2
        return result

    def query(self, x):
        """Query an item."""
        return self.rank(x)
    
    def batch_rank(self, items):
        result = np.zeros(len(items))
        data = items
        for i in range(self.total_levels):
            mask = (items % 2 == 1)
            result[mask] += self.subdomains[i].batch_query(data[mask] - 1)
            data = data // 2
        return result
    
    def batch_query(self, items):
        """Query a pd.Series of values."""
        return self.batch_rank(items)

    def memory_budget(self):
        return self.totalsize
    
    @classmethod
    def combine(cls, dcs_list):
        # Collect everything in first one.
        first = dcs_list[0]
        for j in range(0, first.total_levels):
            sub_cs = [dcs.subdomains[j] for dcs in dcs_list]
            first.subdomains[j] = FasterCountSketch.combine(sub_cs)
        return first
