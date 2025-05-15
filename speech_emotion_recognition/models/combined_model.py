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
                 model_type="combined", # New parameter: "1d", "2d", or "combined"
                 # CNN1D parameters
                 cnn1d_input_channels=1, 
                 # CNN2D parameters
                 cnn2d_input_channels=1, 
                 # Shared/General parameters
                 dropout_rate_cnn=0.3, # Dropout for CNN blocks
                 dropout_rate_mlp=0.5, # Dropout for MLP head
                 # Activation modules can be passed, e.g., nn.SiLU()
                 activation_module_cnn: nn.Module = None, 
                 activation_module_mlp: nn.Module = None
                 ):
        super(CombinedModel, self).__init__()

        self.model_type = model_type.lower()
        if self.model_type not in ["1d", "2d", "combined"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose from '1d', '2d', or 'combined'.")

        # Set default activation modules if not provided
        # Paper summary indicates ReLU for these parts.
        if activation_module_cnn is None:
            # Create a new instance for CNNs, in case it's used elsewhere
            activation_module_cnn = nn.ReLU(inplace=True) 
        if activation_module_mlp is None:
            # Create a new instance for MLP
            activation_module_mlp = nn.ReLU(inplace=True)
            
        # Expected output feature sizes from the individual CNNs
        cnn1d_output_features = 512
        cnn2d_output_features = 1024 

        mlp_input_features = 0

        if self.model_type == "1d" or self.model_type == "combined":
            self.cnn1d = CNN1D(
                input_channels=cnn1d_input_channels, 
                dropout_rate=dropout_rate_cnn,
                activation_module=activation_module_cnn # Pass the instantiated module
            )
            mlp_input_features += cnn1d_output_features
        else:
            self.cnn1d = None

        if self.model_type == "2d" or self.model_type == "combined":
            self.cnn2d = CNN2D(
                input_channels=cnn2d_input_channels,
                dropout_rate=dropout_rate_cnn,
                activation_module=activation_module_cnn # Pass the instantiated module
            )
            mlp_input_features += cnn2d_output_features
        else:
            self.cnn2d = None
        
        if mlp_input_features == 0:
            # This case should not be reached due to the model_type check, but as a safeguard
            raise ValueError("No CNN path selected, MLP input features would be zero.")

        self.fc_block = nn.Sequential(
            nn.Linear(mlp_input_features, 128),
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
        
        features_to_concatenate = []

        if self.model_type == "1d":
            if self.cnn1d is None:
                raise RuntimeError("Model type is '1d' but cnn1d is not initialized.")
            out_1d = self.cnn1d(x_1d) # Expected: (N, 512)
            features_to_concatenate.append(out_1d)
        elif self.model_type == "2d":
            if self.cnn2d is None:
                raise RuntimeError("Model type is '2d' but cnn2d is not initialized.")
            out_2d = self.cnn2d(x_2d) # Expected: (N, 1024)
            features_to_concatenate.append(out_2d)
        elif self.model_type == "combined":
            if self.cnn1d is None or self.cnn2d is None:
                raise RuntimeError("Model type is 'combined' but one or both CNNs are not initialized.")
            out_1d = self.cnn1d(x_1d) # Expected: (N, 512)
            out_2d = self.cnn2d(x_2d) # Expected: (N, 1024)
            features_to_concatenate.append(out_1d)
            features_to_concatenate.append(out_2d)
        else:
            # This case should be caught by the __init__ check
            raise ValueError(f"Invalid model_type '{self.model_type}' in forward pass.")

        if not features_to_concatenate:
             raise RuntimeError("No features were processed by CNNs.")
        
        # Concatenate along the feature dimension (dim=1) if more than one feature set
        if len(features_to_concatenate) > 1:
            processed_features = torch.cat(features_to_concatenate, dim=1)
        else:
            processed_features = features_to_concatenate[0]
            
        output = self.fc_block(processed_features) # Expected: (N, num_classes)
        return output

if __name__ == '__main__':
    batch_size = 4
    crema_num_classes = 6 

    # --- Test with ReLU (default, as per paper summary) ---
    print("--- Testing CombinedModel with ReLU ---")
    
    # Test Combined (default)
    print("Testing model_type='combined'")
    combined_model_relu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="combined" 
        # Default activations (ReLU) will be used
    )
    dummy_x1d = torch.randn(batch_size, 1, 162)
    dummy_x2d = torch.randn(batch_size, 1, 64, 64)
    output_relu = combined_model_relu(dummy_x1d, dummy_x2d)
    print(f"CombinedModel (ReLU, combined) input 1D: {dummy_x1d.shape}")
    print(f"CombinedModel (ReLU, combined) input 2D: {dummy_x2d.shape}")
    print(f"CombinedModel (ReLU, combined) output shape: {output_relu.shape}")
    assert output_relu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_relu.shape}"
    print("CombinedModel (ReLU, combined) test passed.")

    # Test 1D only
    print("\nTesting model_type='1d'")
    model_1d_relu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="1d"
    )
    output_1d_relu = model_1d_relu(dummy_x1d, dummy_x2d) # Still pass both, model should ignore x2d
    print(f"CombinedModel (ReLU, 1d) input 1D: {dummy_x1d.shape}")
    print(f"CombinedModel (ReLU, 1d) input 2D: {dummy_x2d.shape} (should be ignored)")
    print(f"CombinedModel (ReLU, 1d) output shape: {output_1d_relu.shape}")
    assert output_1d_relu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_1d_relu.shape}"
    assert model_1d_relu.cnn1d is not None
    assert model_1d_relu.cnn2d is None
    print("CombinedModel (ReLU, 1d) test passed.")

    # Test 2D only
    print("\nTesting model_type='2d'")
    model_2d_relu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="2d"
    )
    output_2d_relu = model_2d_relu(dummy_x1d, dummy_x2d) # Still pass both, model should ignore x1d
    print(f"CombinedModel (ReLU, 2d) input 1D: {dummy_x1d.shape} (should be ignored)")
    print(f"CombinedModel (ReLU, 2d) input 2D: {dummy_x2d.shape}")
    print(f"CombinedModel (ReLU, 2d) output shape: {output_2d_relu.shape}")
    assert output_2d_relu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_2d_relu.shape}"
    assert model_2d_relu.cnn1d is None
    assert model_2d_relu.cnn2d is not None
    print("CombinedModel (ReLU, 2d) test passed.")


    # --- Test with SiLU (as per user's initial preference) ---
    print("\n--- Testing CombinedModel with SiLU ---")
    activation_cnn_silu = nn.SiLU() 
    activation_mlp_silu = nn.SiLU() 

    # Test Combined with SiLU
    print("Testing model_type='combined' with SiLU")
    combined_model_silu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="combined",
        activation_module_cnn=activation_cnn_silu,
        activation_module_mlp=activation_mlp_silu
    )
    output_silu_combined = combined_model_silu(dummy_x1d, dummy_x2d)
    print(f"CombinedModel (SiLU, combined) output shape: {output_silu_combined.shape}")
    assert output_silu_combined.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_silu_combined.shape}"
    print("CombinedModel (SiLU, combined) test passed.")

    # Test 1D only with SiLU
    print("\nTesting model_type='1d' with SiLU")
    model_1d_silu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="1d",
        activation_module_cnn=nn.SiLU(), # New instance for this model
        activation_module_mlp=nn.SiLU()  # New instance for this model
    )
    output_1d_silu = model_1d_silu(dummy_x1d, dummy_x2d)
    print(f"CombinedModel (SiLU, 1d) output shape: {output_1d_silu.shape}")
    assert output_1d_silu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_1d_silu.shape}"
    assert model_1d_silu.cnn1d is not None
    assert model_1d_silu.cnn2d is None
    print("CombinedModel (SiLU, 1d) test passed.")

    # Test 2D only with SiLU
    print("\nTesting model_type='2d' with SiLU")
    model_2d_silu = CombinedModel(
        num_classes=crema_num_classes,
        model_type="2d",
        activation_module_cnn=nn.SiLU(), # New instance for this model
        activation_module_mlp=nn.SiLU()  # New instance for this model
    )
    output_2d_silu = model_2d_silu(dummy_x1d, dummy_x2d)
    print(f"CombinedModel (SiLU, 2d) output shape: {output_2d_silu.shape}")
    assert output_2d_silu.shape == (batch_size, crema_num_classes), \
        f"Expected output shape ({batch_size}, {crema_num_classes}), got {output_2d_silu.shape}"
    assert model_2d_silu.cnn1d is None
    assert model_2d_silu.cnn2d is not None
    print("CombinedModel (SiLU, 2d) test passed.")

    # Print Model structure
    print("\nCombinedModel (SiLU, combined) structure:")
    print(combined_model_silu)
    total_params = sum(p.numel() for p in combined_model_silu.parameters() if p.requires_grad)
    print(f"Total trainable parameters (SiLU, combined model): {total_params:,}")

    print("\nCombinedModel (ReLU, 1d) structure:")
    print(model_1d_relu)
    total_params_1d = sum(p.numel() for p in model_1d_relu.parameters() if p.requires_grad)
    print(f"Total trainable parameters (ReLU, 1d model): {total_params_1d:,}")

    print("\nCombinedModel (ReLU, 2d) structure:")
    print(model_2d_relu)
    total_params_2d = sum(p.numel() for p in model_2d_relu.parameters() if p.requires_grad)
    print(f"Total trainable parameters (ReLU, 2d model): {total_params_2d:,}") 