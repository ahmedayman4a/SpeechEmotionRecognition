import torch
import torch.nn as nn

try:
    from .cnn_1d import CNN1D
    from .cnn_2d import CNN2D
except ImportError:
    print("Attempting direct import for CNN1D and CNN2D (e.g., for standalone testing).")
    from cnn_1d import CNN1D
    from cnn_2d import CNN2D

class MLPHead(nn.Module):
    def __init__(self, input_size, num_classes, mlp_dropout_rate=0.5, activation_module: nn.Module = None):
        super(MLPHead, self).__init__()
        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(input_size, 128) # Paper: "Dense Layer of 128 units"
        self.act1 = activation_module
        self.dropout = nn.Dropout(mlp_dropout_rate)
        self.fc2 = nn.Linear(128, num_classes) # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, 
                 num_classes=6, 
                 cnn1d_input_channels=1, 
                 cnn1d_num_features_dim=162,
                 cnn1d_initial_out_channels=64,
                 cnn1d_block_channels=[64, 128, 256, 512],
                 cnn1d_output_features=256, 
                 cnn2d_input_channels=1,
                 cnn2d_initial_out_channels=32,
                 cnn2d_block_channels=[32, 64, 128, 256],
                 cnn2d_output_features=512, 
                 cnn_dropout_rate=0.3, 
                 mlp_dropout_rate=0.5, 
                 activation_name='relu'): # 'relu' or 'silu'
        super(CombinedModel, self).__init__()

        if activation_name.lower() == 'relu':
            activation_module = nn.ReLU(inplace=True)
        elif activation_name.lower() == 'silu':
            activation_module = nn.SiLU() # No inplace for SiLU generally
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.cnn1d = CNN1D(
            input_channels=cnn1d_input_channels,
            num_features_input_dim=cnn1d_num_features_dim,
            initial_out_channels=cnn1d_initial_out_channels,
            block_channels=cnn1d_block_channels,
            dropout_rate=cnn_dropout_rate,
            activation_module=activation_module,
            output_feature_size=cnn1d_output_features # Pass this to the new CNN1D
        )

        self.cnn2d = CNN2D(
            input_channels=cnn2d_input_channels,
            initial_out_channels=cnn2d_initial_out_channels,
            block_channels=cnn2d_block_channels,
            dropout_rate=cnn_dropout_rate,
            activation_module=activation_module,
            output_feature_size=cnn2d_output_features # Pass this to the new CNN2D
        )

        # Total features after concatenation
        self.total_features = cnn1d_output_features + cnn2d_output_features
        
        self.mlp_head = MLPHead(
            input_size=self.total_features,
            num_classes=num_classes,
            mlp_dropout_rate=mlp_dropout_rate,
            activation_module=activation_module
        )

    def forward(self, x_1d, x_2d):
        # x_1d: (batch_size, cnn1d_input_channels, cnn1d_num_features_dim) e.g. (N, 1, 162)
        # x_2d: (batch_size, cnn2d_input_channels, H, W_variable) e.g. (N, 1, 64, 188)
        
        features_1d = self.cnn1d(x_1d) # Expected output: (N, cnn1d_output_features)
        features_2d = self.cnn2d(x_2d) # Expected output: (N, cnn2d_output_features)
        
        combined_features = torch.cat((features_1d, features_2d), dim=1) # (N, total_features)
        
        output_logits = self.mlp_head(combined_features)
        return output_logits

if __name__ == '__main__':
    batch_size = 4
    num_classes_test = 6
    cnn1d_feat_len = 162
    cnn2d_h = 64
    cnn2d_w_variable = 188 # Example variable width

    # Config for combined model (using new ResNet based CNNs)
    config_combined = {
        'num_classes': num_classes_test,
        'cnn1d_input_channels': 1,
        'cnn1d_num_features_dim': cnn1d_feat_len,
        'cnn1d_initial_out_channels': 64, # Example, could be from a global config
        'cnn1d_block_channels': [64, 128, 256, 512], # Example
        'cnn1d_output_features': 256, 
        'cnn2d_input_channels': 1,
        'cnn2d_initial_out_channels': 32, # Example
        'cnn2d_block_channels': [32, 64, 128, 256], # Example
        'cnn2d_output_features': 512, 
        'cnn_dropout_rate': 0.2,
        'mlp_dropout_rate': 0.4,
        'activation_name': 'relu'
    }

    print("Testing CombinedModel with ResNet-like backbones (ReLU)...")
    combined_model_relu = CombinedModel(**config_combined)
    
    dummy_input_1d = torch.randn(batch_size, config_combined['cnn1d_input_channels'], config_combined['cnn1d_num_features_dim'])
    dummy_input_2d = torch.randn(batch_size, config_combined['cnn2d_input_channels'], cnn2d_h, cnn2d_w_variable)
    
    output_combined_relu = combined_model_relu(dummy_input_1d, dummy_input_2d)
    
    print(f"CombinedModel (ReLU) input 1D shape: {dummy_input_1d.shape}")
    print(f"CombinedModel (ReLU) input 2D shape: {dummy_input_2d.shape}")
    print(f"CombinedModel (ReLU) output shape: {output_combined_relu.shape}")
    assert output_combined_relu.shape == (batch_size, num_classes_test), \
        f"Expected output shape ({batch_size}, {num_classes_test}), got {output_combined_relu.shape}"
    print("CombinedModel (ReLU) test passed.")

    # Test with SiLU
    config_combined_silu = config_combined.copy()
    config_combined_silu['activation_name'] = 'silu'
    print("\nTesting CombinedModel with ResNet-like backbones (SiLU)...")
    combined_model_silu = CombinedModel(**config_combined_silu)
    output_combined_silu = combined_model_silu(dummy_input_1d, dummy_input_2d)

    print(f"CombinedModel (SiLU) input 1D shape: {dummy_input_1d.shape}")
    print(f"CombinedModel (SiLU) input 2D shape: {dummy_input_2d.shape}")
    print(f"CombinedModel (SiLU) output shape: {output_combined_silu.shape}")
    assert output_combined_silu.shape == (batch_size, num_classes_test), \
        f"Expected output shape ({batch_size}, {num_classes_test}), got {output_combined_silu.shape}"
    print("CombinedModel (SiLU) test passed.")

    # Test with another variable width for 2D input
    dummy_input_2d_v2 = torch.randn(batch_size, config_combined['cnn2d_input_channels'], cnn2d_h, cnn2d_w_variable // 2)
    output_combined_relu_v2 = combined_model_relu(dummy_input_1d, dummy_input_2d_v2)
    print(f"\nCombinedModel (ReLU) input 2D shape (width={cnn2d_w_variable // 2}): {dummy_input_2d_v2.shape}")
    print(f"CombinedModel (ReLU) output shape (width={cnn2d_w_variable // 2}): {output_combined_relu_v2.shape}")
    assert output_combined_relu_v2.shape == (batch_size, num_classes_test), \
        f"Expected output shape ({batch_size}, {num_classes_test}), got {output_combined_relu_v2.shape}"
    print("CombinedModel (ReLU) with different 2D width test passed.") 