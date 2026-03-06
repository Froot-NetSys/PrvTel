import torch

class VisdomLogger:
    """Handles all Visdom visualization logic for training"""
    def __init__(self, vis, is_graph_vae=False):
        """
        Args:
            vis: Visdom visualizer instance
            is_graph_vae: Whether to include GraphVAE specific plots
        """
        self.vis = vis
        self.is_graph_vae = is_graph_vae
        self.windows = {}
        
        if vis is not None:
            self._init_plots()
    
    def _init_plots(self):
        """Initialize all visualization plots"""
        init_x = torch.zeros(1)
        
        # Main loss plot (ELBO, Reconstruction, and KL Divergence)
        self.windows['main_loss'] = self.vis.line(
            X=init_x,
            Y=torch.zeros((1, 3)),
            opts={
                'title': 'Main Training Losses',
                'legend': ['Total', 'Reconstruction', 'Regularization'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'showlegend': True
            }
        )

        # Categorical loss plot
        self.windows['categorical_loss'] = self.vis.line(
            X=init_x,
            Y=torch.zeros(1),
            opts={
                'title': 'Categorical Reconstruction Loss',
                'legend': ['Categorical'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'showlegend': True
            }
        )

        # Numerical loss plot
        self.windows['numerical_loss'] = self.vis.line(
            X=init_x,
            Y=torch.zeros(1),
            opts={
                'title': 'Numerical Reconstruction Loss',
                'legend': ['Numerical'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'showlegend': True
            }
        )

        if self.is_graph_vae:
            self.windows['graph_loss'] = self.vis.line(
                X=init_x,
                Y=torch.zeros((1, 2)),
                opts={
                    'title': 'Graph VAE Losses',
                    'legend': ['Latent Graph', 'Top-K'],
                    'xlabel': 'Epoch',
                    'ylabel': 'Loss',
                    'showlegend': True
                }
            )

    def update(self, epoch, metrics, logger):
        """Update all plots with new metrics
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metric values
            logger: Logger instance for error reporting
        """
        if self.vis is None:
            return

        epoch_data = torch.tensor([epoch])
        
        try:
            # Update main losses plot (now including KL divergence)
            main_loss_data = torch.tensor([[
                metrics['train_loss'],
                metrics['reconstruct_loss'],
                metrics['regularization_loss']
            ]], dtype=torch.float32)
            
            self._update_plot('main_loss', epoch_data, main_loss_data, epoch)

            # Update categorical loss plot
            cat_data = torch.tensor([[
                metrics['categorical_reconstruct']
            ]], dtype=torch.float32)
            
            self._update_plot('categorical_loss', epoch_data, cat_data, epoch)

            # Update numerical loss plot
            num_data = torch.tensor([[
                metrics['numerical_reconstruct']
            ]], dtype=torch.float32)
            
            self._update_plot('numerical_loss', epoch_data, num_data, epoch)
            
            if self.is_graph_vae:
                graph_data = torch.tensor([[
                    metrics['latent_graph_loss'],
                    metrics['top_k_loss']
                ]], dtype=torch.float32)
                
                self._update_plot('graph_loss', epoch_data, graph_data, epoch)

        except Exception as e:
            logger.error(f"Failed to update Visdom plot: {e}")

    def _update_plot(self, window_name, x_data, y_data, epoch):
        """Helper method to update a single plot
        
        Args:
            window_name: Name of the window to update
            x_data: X-axis data
            y_data: Y-axis data
            epoch: Current epoch number
        """
        self.vis.line(
            X=x_data,
            Y=y_data,
            win=self.windows[window_name],
            update='append' if epoch > 0 else 'replace'
        )