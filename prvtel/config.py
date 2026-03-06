import os
import torch
from datetime import datetime


class Config():
    """
    VAE model training configs
    batch_size is the most important hyperparameter to tune. for synthetic data, batch_size=10 is good.
    for synthetic data, pre_proc_emthod does not matter much.
    """
    # Default values as class constants
    DEFAULT_PRE_PROC_METHOD = 'GMM'
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_N_EPOCHS = 100
    DEFAULT_LOGGING_FREQ = 1
    DEFAULT_PATIENCE = 50
    DEFAULT_DELTA = 10.0
    DEFAULT_USE_GRAPH_VAE = False

    # sampling parameters
    DEFAULT_NON_UNIFORM_SAMPLING = False # backwards compatibility 
    DEFAULT_SHUFFLE_BATCHES = False

    # Privacy defaults
    DEFAULT_DIFFERENTIAL_PRIVACY = False
    DEFAULT_SAMPLE_RATE = 0.1
    DEFAULT_C = 1e16
    DEFAULT_NOISE_SCALE = 0.25
    DEFAULT_TARGET_EPS = 2.0
    DEFAULT_TARGET_DELTA = 1e-5
    
    # Path defaults
    DEFAULT_FILE_FORMAT = 'csv'  # 'csv' or 'parquet'
    DEFAULT_PREPROCESSOR_PATH = '../data/preprocessed/preprocessors.pkl'
    DEFAULT_PREPROCESSED_DATA_PATH = None
    DEFAULT_USE_PREPROCESSED = False

    # Big data defaults
    DEFAULT_BLOCKSIZE = 'auto'
    DEFAULT_NUM_CHUNKS_CACHED = None

    # Preprocessing defaults
    DEFAULT_EXCLUDED_COLUMNS = ['time', 'timestamp']
    DEFAULT_CATEGORICALS = []

    def __init__(self):
        # Only initialize non-argument related setup here
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create required directories
        self.model_save_dir = os.path.join(self.project_dir, 'saved_model/')
        self.results_dir = os.path.join(self.project_dir, 'results/')
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Set timestamp for model saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{timestamp}.pth')

        # Initialize with class defaults
        self.batch_size = self.DEFAULT_BATCH_SIZE
        self.n_epochs = self.DEFAULT_N_EPOCHS
        self.logging_freq = self.DEFAULT_LOGGING_FREQ
        self.patience = self.DEFAULT_PATIENCE
        self.delta = self.DEFAULT_DELTA
        
        # Privacy params
        self.differential_privacy = self.DEFAULT_DIFFERENTIAL_PRIVACY
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.C = self.DEFAULT_C
        self.noise_scale = self.DEFAULT_NOISE_SCALE
        self.target_eps = self.DEFAULT_TARGET_EPS
        self.target_delta = self.DEFAULT_TARGET_DELTA
        
        # Paths
        self.input_data_path = None  # Required, no default
        self.config_file_path = None # Required, no default
        self.preprocessor_path = self.DEFAULT_PREPROCESSOR_PATH
        self.preprocessed_data_path = self.DEFAULT_PREPROCESSED_DATA_PATH
        self.adjacency_matrix_path = None
        self.use_preprocessed = self.DEFAULT_USE_PREPROCESSED
        self.file_format = self.DEFAULT_FILE_FORMAT
        
        # Graph VAE params
        self.use_graph_vae = self.DEFAULT_USE_GRAPH_VAE

        # Big data params
        self.blocksize = self.DEFAULT_BLOCKSIZE
        self.num_chunks_cached = self.DEFAULT_NUM_CHUNKS_CACHED

        # Preprocessing
        self.excluded_columns = self.DEFAULT_EXCLUDED_COLUMNS
        self.categoricals = self.DEFAULT_CATEGORICALS
        self.pre_proc_method = self.DEFAULT_PRE_PROC_METHOD

        # Sampling params
        self.non_uniform_sampling = self.DEFAULT_NON_UNIFORM_SAMPLING
        self.shuffle_batches = self.DEFAULT_SHUFFLE_BATCHES

        self.use_visdom = False
        self.visdom_server = 'http://localhost'
        self.visdom_port = 8097


    @classmethod
    def from_args(cls, args):
        config = cls()  # Create instance without arguments
        # Set attributes from args
        for key, value in vars(args).items():
            if value is not None:
                setattr(config, key, value)
        return config

    @staticmethod
    def add_arguments(parser):
        """
        Add model-specific arguments to the parser.
        All default values are pulled from class constants.
        """
        parser.add_argument('--batch_size', type=int, default=Config.DEFAULT_BATCH_SIZE,
                          help='Batch size for training. For synthetic data, batch_size=10 is good')
        parser.add_argument('--n_epochs', type=int, default=Config.DEFAULT_N_EPOCHS,
                          help='Number of training epochs')
        parser.add_argument('--logging_freq', type=int, default=Config.DEFAULT_LOGGING_FREQ,
                          help='Number of epochs between logging')
        parser.add_argument('--patience', type=int, default=Config.DEFAULT_PATIENCE,
                          help='Early stopping patience')
        parser.add_argument('--delta', type=float, default=Config.DEFAULT_DELTA,
                          help='Improvement threshold for early stopping')
        
        # Privacy params
        parser.add_argument('--differential_privacy', 
                            action='store_true', 
                            default=Config.DEFAULT_DIFFERENTIAL_PRIVACY,
                            help='Enable differential privacy')
        parser.add_argument('--sample_rate', type=float, default=Config.DEFAULT_SAMPLE_RATE,
                          help='Sampling rate for DP')
        parser.add_argument('--C', type=float, default=Config.DEFAULT_C,
                          help='Clipping threshold for DP')
        parser.add_argument('--noise_scale', type=float, default=Config.DEFAULT_NOISE_SCALE,
                          help='Noise multiplier for DP')
        parser.add_argument('--target_eps', type=float, default=Config.DEFAULT_TARGET_EPS,
                          help='Target epsilon for privacy accountant')
        parser.add_argument('--target_delta', type=float, default=Config.DEFAULT_TARGET_DELTA,
                          help='Target delta for privacy accountant')
        
        # Paths
        parser.add_argument('--input_data_path', type=str, required=True,
                          help='Path to input data file')
        parser.add_argument('--config_file_path', type=str, required=True,
                            help='JSON file that specifies model configuration. '
                            'Optionally specify how to transform different columns as well.')
        parser.add_argument('--model_save_path', type=str,
                          help='Override path to save model checkpoint')
        parser.add_argument('--preprocessor_path', type=str,
                          default=Config.DEFAULT_PREPROCESSOR_PATH,
                          help='Path to save preprocessors and metadata needed for inference. Should be a .pkl file.')
        parser.add_argument('--preprocessed_data_path', type=str,
                          default=Config.DEFAULT_PREPROCESSED_DATA_PATH,
                          help='Path to save processed data and related metadata.')
        parser.add_argument("--adjacency_matrix_path", type=str,
                            default="../data/preprocessed/adjacency_matrix.npy",
                            help="Path to save/load the adjacency matrix")
        parser.add_argument('--results_dir', type=str,
                          help='Directory to save results')
        parser.add_argument('--file_format', type=str, choices=['csv', 'parquet'],
                          default=Config.DEFAULT_FILE_FORMAT,
                          help='Format of input data file (csv or parquet). Default is csv.')
        
        # TODO: for any boolean flags, even if we pass it as False, it will be set as True.
        # Graph VAE params
        parser.add_argument('--use_graph_vae',
                            action='store_true',
                            default=Config.DEFAULT_USE_GRAPH_VAE,
                            help='Use graph-based VAE architecture')
        
        # Big data params
        parser.add_argument('--blocksize', type=str, default=Config.DEFAULT_BLOCKSIZE,
                          help='Size of the chunks to break up the data into (e.g "4MB")')
        parser.add_argument('--num_chunks_cached', type=int, default=Config.DEFAULT_NUM_CHUNKS_CACHED,
                          help='Number of chunks to persist in RAM at a time.')
        
        # Preprocessing
        parser.add_argument('--excluded_columns', nargs='+', type=str, default=Config.DEFAULT_EXCLUDED_COLUMNS,
                          help='Columns to drop from the input data.')
        parser.add_argument('--categoricals', type=str, nargs='*', default=Config.DEFAULT_CATEGORICALS,
                          help='Space-separated list of categorical columns (or "auto" to infer them). Example: --categoricals col1 col2')
        parser.add_argument('--pre_proc_method', type=str, default=Config.DEFAULT_PRE_PROC_METHOD,
                            help='Preprocessing method to use (e.g. "GMM"). Used when "transforms" not provided in config JSON file ')
        parser.add_argument('--use_preprocessed', 
                            action='store_true', 
                            default=Config.DEFAULT_USE_PREPROCESSED,
                            help='Use preprocessed data instead of raw data')
        
        # Visdom
        parser.add_argument('--use_visdom', action='store_true', default=False,
                          help='Enable Visdom visualization')
        parser.add_argument('--visdom_server', type=str, default='http://localhost',
                          help='Visdom server address')
        parser.add_argument('--visdom_port', type=int, default=8097,
                          help='Visdom server port')
        
        # Add new sampling-related arguments
        parser.add_argument('--non_uniform_sampling', 
                            action='store_true', 
                            default=Config.DEFAULT_NON_UNIFORM_SAMPLING,
                            help='Turns off uniform sampling')
        parser.add_argument('--shuffle_batches', 
                            action='store_true', 
                            default=Config.DEFAULT_SHUFFLE_BATCHES,
                            help='Whether to shuffle batches when not using uniform sampling')
        
        return parser

if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
