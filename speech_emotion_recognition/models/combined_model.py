import torch
import torch.nn as nn

try:
    from .cnn_1d import CNN1D
    from .cnn_2d import CNN2D
except ImportError:
    print("Attempting direct import for CNN1D and CNN2D (e.g., for standalone testing).")
    from cnn_1d import CNN1D
    from cnn_2d import CNN2D

class CombinedModel(nn.Module):
    def __init__(self, 
                 num_classes=6, # Default to 6 for CREMA-D emotions
                 # CNN1D parameters
                 cnn1d_input_channels=1, 
                 cnn1d_num_features_input_dim=162,
                 # CNN2D parameters
                 cnn2d_input_channels=1, 
                 cnn2d_img_height=64, 
                 cnn2d_img_width=64,
                 # Shared/General parameters
                 dropout_rate_cnn=0.3, # Dropout for CNN blocks
                 dropout_rate_mlp=0.5, # Dropout for MLP head
                 # Activation modules can be passed, e.g., nn.SiLU()
                 activation_module_cnn: nn.Module = None, 
                 activation_module_mlp: nn.Module = None
                 ):
        super(CombinedModel, self).__init__()

        # Set default activation modules if not provided
        # Paper summary indicates ReLU for these parts.
        if activation_module_cnn is None:
            # Create a new instance for CNNs, in case it's used elsewhere
            activation_module_cnn = nn.ReLU(inplace=True) 
        if activation_module_mlp is None:
            # Create a new instance for MLP
            activation_module_mlp = nn.ReLU(inplace=True)
            
        self.cnn1d = CNN1D(
            input_channels=cnn1d_input_channels, 
            dropout_rate=dropout_rate_cnn,
            activation_module=activation_module_cnn # Pass the instantiated module
        )
        # Expected output of cnn1d is 512 features after GAP

        self.cnn2d = CNN2D(
            input_channels=cnn2d_input_channels,
            dropout_rate=dropout_rate_cnn,
            activation_module=activation_module_cnn # Pass the instantiated module
        )
        # Expected output of cnn2d is 1024 features after GAP
        
        # Flattened feature sizes from the paper (and derived model architecture)
        cnn1d_output_features = 512
        cnn2d_output_features = 1024 
        concatenated_features = cnn1d_output_features + cnn2d_output_features

        self.fc_block = nn.Sequential(
            nn.Linear(concatenated_features, 128),
            nn.LayerNorm(128),
            activation_module_mlp, # Use the instantiated MLP activation module
            nn.Dropout(dropout_rate_mlp),
            nn.Linear(128, num_classes) 
        )

    def forward(self, x_1d, x_2d):
        """
        Args:
            x_1d (torch.Tensor): Input for the 1D CNN branch. 
                                 Shape: (batch_size, cnn1d_input_channels, cnn1d_num_features_input_dim)
                                 Example: (N, 1, 162)
            x_2d (torch.Tensor): Input for the 2D CNN branch. 
                                 Shape: (batch_size, cnn2d_input_channels, cnn2d_img_height, cnn2d_img_width)
                                 Example: (N, 1, 64, 64)
        Returns:
            torch.Tensor: Output logits for each class. Shape: (batch_size, num_classes)
        """
        out_1d = self.cnn1d(x_1d) # Expected: (N, 512)
        out_2d = self.cnn2d(x_2d) # Expected: (N, 1024)
        
        # Concatenate along the feature dimension (dim=1)
        concatenated = torch.cat((out_1d, out_2d), dim=1)
        
        output = self.fc_block(concatenated) # Expected: (N, num_classes)
        return output

if __name__ == '__main__':
    batch_size = 4
    crema_num_classes = 6 

    # --- Test with ReLU (default, as per paper summary) ---
    print("--- Testing CombinedModel with ReLU ---")
    combined_model_relu = CombinedModel(
        num_classes=crema_num_classes,
        # Default activations (ReLU) will be used
    )
    dummy_x1d_relu = torch.randn(batch_size, 1, 162)
    dummy_x2d_relu = torch.randn(batch_size, 1, 64, 64)
    output_relu = combined_model_relu(dummy_x1d_relu, dummy_x2d_relu)
    print(f"CombinedModel (ReLU) input 1D: {dummy_x1d_relu.shape}")
    print(f"CombinedModel (ReLU) input 2D: {dummy_x2d_relu.shape}")
    print(f"CombinedModel (ReLU) output shape: {output_relu.shape}")
    assert output_relu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_relu.shape}"
    print("CombinedModel (ReLU) test passed.")

    # --- Test with SiLU (as per user's initial preference) ---
    print("\n--- Testing CombinedModel with SiLU ---")
    activation_cnn_silu = nn.SiLU() # Create SiLU instance
    activation_mlp_silu = nn.SiLU() # Create SiLU instance
    combined_model_silu = CombinedModel(
        num_classes=crema_num_classes,
        activation_module_cnn=activation_cnn_silu,
        activation_module_mlp=activation_mlp_silu
    )
    dummy_x1d_silu = torch.randn(batch_size, 1, 162)
    dummy_x2d_silu = torch.randn(batch_size, 1, 64, 64)
    output_silu = combined_model_silu(dummy_x1d_silu, dummy_x2d_silu)
    print(f"CombinedModel (SiLU) input 1D: {dummy_x1d_silu.shape}")
    print(f"CombinedModel (SiLU) input 2D: {dummy_x2d_silu.shape}")
    print(f"CombinedModel (SiLU) output shape: {output_silu.shape}")
    assert output_silu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_silu.shape}"
    print("CombinedModel (SiLU) test passed.")

    # You can also print the model structure or number of parameters
    # print("\nCombinedModel (SiLU) structure:")
    # print(combined_model_silu)
    # total_params = sum(p.numel() for p in combined_model_silu.parameters() if p.requires_grad)
    # print(f"Total trainable parameters (SiLU model): {total_params:,}") 