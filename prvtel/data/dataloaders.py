import dask.dataframe as dd
import torch
import queue
import math
import threading


class DataFrameIter:
    """Ripped straight from core.merlin.io. Iterates through partitions of dask DataFrame."""
    
    def __init__(self, ddf, columns=None, indices=None, partition_lens=None):
        self.indices = indices if isinstance(indices, list) else list(range(ddf.npartitions))
        self.ddf = ddf
        self.columns = columns
        self.partition_lens = partition_lens if partition_lens else [None] * self.ddf.npartitions
        self.length = None

    def __call__(self, indices):
        """Sets the indices to iterate over. Length will have to be recomputed though."""
        self.indices = indices
        self.length = None

    def __len__(self):
        """Caches length computation. Assumes that underlying dataframe won't change."""
        if self.length:
            return self.length
        
        # Check that every partition has a length.
        part_lens = [self.partition_lens[i] for i in self.indices if self.partition_lens[i] is not None]
        if len(part_lens) == len(self.indices):
            # Assumes that length won't change somehow.
            self.length = sum(part_lens[i] for i in self.indices)
            return self.length
        # Computing length manually.
        if len(self.indices) < self.ddf.npartitions:
            self.length = len(self.ddf.partitions[self.indices])
            return self.length
        
        self.length = len(self.ddf)
        return self.length

    def __iter__(self):
        # Compute length and partition lengths while iterating.
        length = 0
        for i in self.indices:
            part = self.ddf.partitions[i]
            if self.columns:
                result = part[self.columns].compute()
            else:
                result = part.compute()

            self.partition_lens[i] = len(result)
            length += self.partition_lens[i]
            yield result

            # Is this here to make sure part gets GC'd?
            part = None
            result = None

        self.length = length


class PartitionRing(DataFrameIter):
    """
    Maintains a FIFO queue of partitions. When first initialized as an iterator, will eagerly
    compute the first n partitions (specified by cache_size) and keep them in RAM. Whenever a
    chunk is popped from the queue, the next chunk after the last one in the queue begins computing
    in a sliding window fashion.
    """

    def __init__(self, ddf, columns=None, indices=None, partition_lens=None, cache_size=None):
        super().__init__(ddf, columns, indices, partition_lens)
        self.part_cache = [None] * ddf.npartitions
    
        # None = just cache (persist) everything.
        self.cache_size = len(self.indices) if cache_size is None else cache_size

        self.initialized = False

    def __call__(self, indices):
        # If already initialized, just use existing cache to build sharded cache.
        if self.initialized:
            new_cache = [None] * self.ddf.npartitions
            num_parts = 0
            # Only include computed partitions that are part of the split.
            for i in indices:
                part = self.part_cache[i]
                if part is not None:
                    num_parts += 1
                new_cache[i] = part

            # Limit cache size to the split obtained as well. This should work
            # since each shard interleaves chunks.
            self.cache_size = num_parts
            self.num_cached = num_parts
            self.end = num_parts - 1

            self.part_cache = new_cache
        else:
            # Scale by change in number of partitions.
            ratio = len(indices) / len(self.indices)
            self.cache_size = round(self.cache_size * ratio)

        # Change to new indices and clear length calculation (like in regular DataFrameIter).
        super().__call__(indices)

    def __iter__(self):
        if not self.initialized:
            self._init_cache(self.cache_size)
        # Compute length and partition lengths while iterating.
        length = 0
        for i in self.indices:
            # Get computing partition from cache.
            part = self.part_cache[i]

            # This should only happen if cache size is 0...
            if part is None:
                part = self.ddf.partitions[i]

            # Compute like before.
            if self.columns:
                result = part[self.columns].compute()
            else:
                result = part.compute()

            # Shift sliding window if only caching fraction of data.
            if 0 < self.cache_size < len(self.indices):
                self.part_cache[i] = None
                self._cache_next()

            # Cache length like before.
            self.partition_lens[i] = len(result)
            length += self.partition_lens[i]
            yield result

            # Is this here to make sure part gets GC'd?
            part = None
            result = None

        self.length = length

    def _init_cache(self, cache_size):
        # Eagerly start computing the first few partitions.
        for i in self.indices[:cache_size]:
            self.part_cache[i] = self.ddf.partitions[i].persist()

        # Number of persisting parts (not sure if needed).
        self.num_cached = len(self.indices[:cache_size])
        # Index in self.indices of the last partition in the queue.
        self.end = self.num_cached - 1

        self.initialized = True

    def _cache_next(self):
        """Start persisting (caching) the next partition in the queue."""
        # Move self.indices pointer to the right.
        self.end = (self.end + 1) % len(self.indices)
        # Get next part and cache.
        part_i = self.indices[self.end]
        part = self.ddf.partitions[part_i].persist()
        self.part_cache[part_i] = part
        

class ChunkDataset(torch.utils.data.IterableDataset):
    """Makes batches of PyTorch tensors (GPU) out of dask_cudf.DataFrame partitions."""

    def __init__(
        self,
        ddf,
        batch_size=1024,
        dtype=None,
        keep_spill=True,
        device=None,
        cache_size=None
    ):
        self.ddf = ddf
        self.batch_size = batch_size
        # Default should be float32.
        self.dtype = dtype if dtype else torch.get_default_dtype()

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.data = PartitionRing(ddf, cache_size=cache_size)
        # Save the original indices of the data.
        self.indices = self.data.indices

        self.keep_spill = keep_spill

    def __len__(self):
        """
        Get the number of batches in the dataloader explicitly. May trigger an expensive
        computation, so calling this is not advised, but we have to have this for
        compatibility with Opacus.
        """
        num_batches = math.ceil(len(self.data) / self.batch_size)
        return num_batches

    def __iter__(self):
        batch_iterator = self.get_batches()
        for batch_group in batch_iterator:
            for batch in batch_group:
                # Need to put in iterable/tuple because vae.train() unpacks output of data loader.
                yield (batch,)

    def create_worker_split(self, rank, world_size):
        """
        Shards underlying Dask DataFrame for individual worker
        processes by assigning each worker a roughly even amount of chunks.
        Should only be called when doing distributed training, i.e. within a 
        multiprocess context where the default process group has been initialized.
        """
        indices = self.indices
        # Interleave chunks between workers, so worker 0 gets every 1st chunk, worker 1,
        # every 2nd chunk, etc. This preserves sequential order (kind of).
        self.data(indices[rank::world_size])
        return self

    def get_batches(self):
        """
        A generator that turns each cuDF partition into a list of torch.Tensor batches.
        Assumes that self.data was initialized already.
        """
        spill: torch.Tensor = None
        for chunk in self.data:
            chunk_tensor = self.df_to_tensor(chunk)
            if spill is not None and spill.numel() > 0:
                chunk_tensor = torch.concat([spill, chunk_tensor])
            batches, spill = self.batch_tensors(chunk_tensor)
            if batches:
                yield batches
            chunk = None
            chunk_tensor = None
            batches = None
        # Emit spillover.
        if spill is not None:
            yield [spill]


    def df_to_tensor(self, chunk):
        df_arr = chunk.values
        tensor = torch.as_tensor(df_arr, device=self.device, dtype=self.dtype)
        return tensor
    
    def batch_tensors(self, chunk_tensor):
        """Splits larger tensor into list of batches. Creates some spill if keep_spill = True."""
        batches = list(torch.split(chunk_tensor, split_size_or_sections=self.batch_size))
        spill = None
        if len(batches) > 0:
            if batches[-1].shape[0] < self.batch_size:
                # Have to clone otherwise spill will eat memory (?).
                if self.keep_spill:
                    spill = batches[-1].clone()
                batches = batches[:-1]
        return batches, spill
    

class ThreadedChunkDataset(ChunkDataset):
    """Uses threads to prefetch partitions (and convert them to tensors) in the background."""

    def __init__(
            self, 
            ddf, 
            batch_size=1024, 
            dtype=None, 
            keep_spill=True,
            device=None,
            cache_size=None, 
            qsize=1
        ):
        super().__init__(ddf, batch_size, dtype, keep_spill, device, cache_size)
        self.batch_queue = queue.Queue(qsize)
        self.stop_event = threading.Event()
        self.thread = None
        self.batch_group = None

    def __iter__(self):
        # if self.create_worker_split():
        #     self.device = 'cpu'
        # I'm assuming this start stop stuff is for if it gets reinitialized before it
        # finishes elsewhere?
        self.stop()
        if self.stopped:
            self.start()

        # Start prefetching chunks and converting into batches.
        t = threading.Thread(target=self.load_batches)
        t.daemon = True
        t.start()
        self.thread = t

        while True:
            # Only proceed if there is data to be fetched.
            # Prevents from blocking on empty data.
            if not self.working and self.empty:
                self.thread = None
                self.batch_group = None
                return
            batch_group = self.dequeue()
            for batch in batch_group:
                yield (batch,)
            batch_group = None
            
    def dequeue(self):
        chunks = self.batch_queue.get()
        if isinstance(chunks, Exception):
            self.stop()
            raise chunks
        return chunks

    def enqueue(self, packet):
        while True:
            if self.stopped:
                return True
            try:
                self.batch_queue.put(packet, timeout=1e-6)
                return False
            except queue.Full:
                continue

    def load_batches(self):
        try:
            self.enqueue_batches()
        except Exception as e:  # pylint: disable=broad-except
            self.enqueue(e)

    def enqueue_batches(self):
        """
        A generator that turns each cuDF partition into a list of torch.Tensor batches.
        Assumes that self.data was initialized already.
        """
        for chunk_batch in self.get_batches():
            if self.stopped:
                return
            if len(chunk_batch) > 0:
                # put returns True if buffer is stopped before
                # packet can be put in queue. Keeps us from
                # freezing on a put on a full queue
                if self.enqueue(chunk_batch):
                    return
            # Does this free memory?
            chunk_batch = None

    @property
    def stopped(self):
        return self.stop_event.is_set()
    
    @property
    def working(self):
        if self.thread is not None:
            return self.thread.is_alive()
        return False
    
    @property
    def empty(self):
        return self.batch_queue.empty()

    def stop(self):
        if self.thread is not None:
            if not self.stopped:
                # Stop thread.
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.batch_queue.queue.clear()

    def start(self):
        self.stop_event.clear()


class SequenceChunkDataset(ChunkDataset):
    """
    Dataset that creates temporal sequences from network traffic data.
    Converts individual data points into sequences for RNN-VAE training.
    """

    def __init__(
        self,
        ddf,
        batch_size=64,  # Smaller batch size for sequences
        seq_len=50,     # Length of each sequence
        stride=1,       # Stride between sequences
        dtype=None,
        keep_spill=True,
        device=None,
        cache_size=None
    ):
        super().__init__(ddf, batch_size, dtype, keep_spill, device, cache_size)
        self.seq_len = seq_len
        self.stride = stride
        
    def get_batches(self):
        """
        A generator that turns each cuDF partition into sequences and then batches.
        """
        spill: torch.Tensor = None
        accumulated_data = None
        
        for chunk in self.data:
            chunk_tensor = self.df_to_tensor(chunk)
            
            # Accumulate data across chunks to form longer sequences
            if accumulated_data is not None:
                accumulated_data = torch.cat([accumulated_data, chunk_tensor], dim=0)
            else:
                accumulated_data = chunk_tensor
            
            # Extract sequences from accumulated data
            sequences = self._create_sequences(accumulated_data)
            
            if sequences is not None and len(sequences) > 0:
                # Add any spill from previous iteration
                if spill is not None and spill.numel() > 0:
                    sequences = torch.cat([spill, sequences], dim=0)
                
                # Create batches from sequences
                batches, spill = self.batch_sequence_tensors(sequences)
                if batches:
                    yield batches
            
            # Keep some data for next iteration to maintain continuity
            if accumulated_data.shape[0] > self.seq_len * 2:
                # Keep the last seq_len samples for continuity
                accumulated_data = accumulated_data[-self.seq_len:]
            
            chunk = None
            chunk_tensor = None
            sequences = None
            batches = None
        
        # Emit final spillover
        if spill is not None:
            yield [spill]
    
    def _create_sequences(self, data):
        """
        Create sequences from data tensor.
        data shape: (num_samples, feature_dim)
        Returns: (num_sequences, seq_len, feature_dim)
        """
        if data.shape[0] < self.seq_len:
            return None
        
        sequences = []
        for start_idx in range(0, data.shape[0] - self.seq_len + 1, self.stride):
            end_idx = start_idx + self.seq_len
            sequence = data[start_idx:end_idx]  # Shape: (seq_len, feature_dim)
            sequences.append(sequence.unsqueeze(0))  # Add batch dimension: (1, seq_len, feature_dim)
        
        if sequences:
            # Concatenate along batch dimension: (num_sequences, seq_len, feature_dim)
            result = torch.cat(sequences, dim=0)
            return result
        return None
    
    def batch_sequence_tensors(self, sequences_tensor):
        """
        Splits sequence tensor into batches while preserving 3D structure.
        sequences_tensor shape: (num_sequences, seq_len, feature_dim)
        Returns batches of shape: (batch_size, seq_len, feature_dim)
        """
        batches = list(torch.split(sequences_tensor, split_size_or_sections=self.batch_size))
        spill = None
        
        if len(batches) > 0:
            if batches[-1].shape[0] < self.batch_size:
                if self.keep_spill:
                    spill = batches[-1].clone()
                batches = batches[:-1]
                
        return batches, spill


class ThreadedSequenceDataset(SequenceChunkDataset):
    """Threaded version of SequenceChunkDataset for better performance."""

    def __init__(
            self, 
            ddf, 
            batch_size=64, 
            seq_len=50,
            stride=1,
            dtype=None, 
            keep_spill=True,
            device=None,
            cache_size=None, 
            qsize=1
        ):
        # Initialize parent class
        ChunkDataset.__init__(self, ddf, batch_size, dtype, keep_spill, device, cache_size)
        self.seq_len = seq_len
        self.stride = stride
        
        # Initialize threading components
        self.batch_queue = queue.Queue(qsize)
        self.stop_event = threading.Event()
        self.thread = None
        self.batch_group = None

    def __iter__(self):
        self.stop()
        if self.stopped:
            self.start()

        # Start prefetching sequences and converting into batches
        t = threading.Thread(target=self.load_batches)
        t.daemon = True
        t.start()
        self.thread = t

        while True:
            if not self.working and self.empty:
                self.thread = None
                self.batch_group = None
                return
            batch_group = self.dequeue()
            for batch in batch_group:
                yield (batch,)
            batch_group = None
            
    def dequeue(self):
        chunks = self.batch_queue.get()
        if isinstance(chunks, Exception):
            self.stop()
            raise chunks
        return chunks

    def enqueue(self, packet):
        while True:
            if self.stopped:
                return True
            try:
                self.batch_queue.put(packet, timeout=1e-6)
                return False
            except queue.Full:
                continue

    def load_batches(self):
        try:
            self.enqueue_batches()
        except Exception as e:
            self.enqueue(e)

    def enqueue_batches(self):
        for chunk_batch in self.get_batches():
            if self.stopped:
                return
            if len(chunk_batch) > 0:
                if self.enqueue(chunk_batch):
                    return
            chunk_batch = None

    @property
    def stopped(self):
        return self.stop_event.is_set()
    
    @property
    def working(self):
        if self.thread is not None:
            return self.thread.is_alive()
        return False
    
    @property
    def empty(self):
        return self.batch_queue.empty()

    def stop(self):
        if self.thread is not None:
            if not self.stopped:
                self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.batch_queue.queue.clear()

    def start(self):
        self.stop_event.clear()