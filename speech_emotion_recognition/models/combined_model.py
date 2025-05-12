import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
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
                 num_classes: int,
                 cnn1d_input_channels: int = 1,
                 cnn1d_num_features_dim: int = 162,
                 cnn1d_initial_out_channels: int = 64,
                 cnn2d_input_channels: int = 1,
                 cnn2d_initial_out_channels: int = 64,
                 cnn_dropout_rate: float = 0.3,
                 mlp_hidden_units: int = 128,
                 mlp_dropout_rate: float = 0.5,
                 activation_name: str = 'relu',
                 layers: List[int] = [2, 2, 2, 2]): 
        super(CombinedModel, self).__init__()

        if activation_name.lower() == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            activation_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.cnn2d = CNN2D(
            input_channels=cnn2d_input_channels,
            initial_out_channels=cnn2d_initial_out_channels,
            layers=layers,
            activation_name=activation_name,
            dropout_rate=cnn_dropout_rate
        )

        cnn2d_output_dim = self.cnn2d.output_dim
        print(f"CNN2D-Only Model: CNN2D out={cnn2d_output_dim}")

        self.mlp_head = nn.Sequential(
            nn.Linear(cnn2d_output_dim, mlp_hidden_units),
            nn.GroupNorm(1, mlp_hidden_units), 
            activation_fn,
            nn.Dropout(mlp_dropout_rate),
            nn.Linear(mlp_hidden_units, num_classes)
        )

    def forward(self, x_2d):
        features_2d = self.cnn2d(x_2d)
        output = self.mlp_head(features_2d)
        return output

if __name__ == '__main__':
    batch_size = 4
    num_classes_test = 6
    cnn2d_h = 64
    cnn2d_w_variable = 188

    config_model = {
        'num_classes': num_classes_test,
        'cnn2d_input_channels': 1,
        'cnn2d_initial_out_channels': 64,
        'cnn_dropout_rate': 0.2,
        'mlp_hidden_units': 128,
        'mlp_dropout_rate': 0.4,
        'activation_name': 'relu'
    }

    print("Testing CNN2D-Only Model (ReLU)...")
    model_relu = CombinedModel(**config_model)
    
    dummy_input_2d = torch.randn(batch_size, config_model['cnn2d_input_channels'], 64, 64)
    
    output_relu = model_relu(dummy_input_2d)
    
    print(f"CNN2D-Only Model (ReLU) input 2D shape: {dummy_input_2d.shape}")
    print(f"CNN2D-Only Model (ReLU) output shape: {output_relu.shape}")
    assert output_relu.shape == (batch_size, config_model['num_classes']), \
        f"Expected output shape ({batch_size}, {config_model['num_classes']}), got {output_relu.shape}"
    print("CNN2D-Only Model (ReLU) test passed.")

    config_model_silu = config_model.copy()
    config_model_silu['activation_name'] = 'silu'
    print("\nTesting CNN2D-Only Model (SiLU)...")
    model_silu = CombinedModel(**config_model_silu)
    output_silu = model_silu(dummy_input_2d)

    print(f"CNN2D-Only Model (SiLU) input 2D shape: {dummy_input_2d.shape}")
    print(f"CNN2D-Only Model (SiLU) output shape: {output_silu.shape}")
    assert output_silu.shape == (batch_size, config_model['num_classes']), \
        f"Expected output shape ({batch_size}, {config_model['num_classes']}), got {output_silu.shape}"
    print("CNN2D-Only Model (SiLU) test passed.")

    dummy_input_2d_v2 = torch.randn(batch_size, config_model['cnn2d_input_channels'], 64, 32)
    output_relu_v2 = model_relu(dummy_input_2d_v2)
    print(f"\nCNN2D-Only Model (ReLU) input 2D shape (width={32}): {dummy_input_2d_v2.shape}")
    print(f"CNN2D-Only Model (ReLU) output shape (width={32}): {output_relu_v2.shape}")
    assert output_relu_v2.shape == (batch_size, config_model['num_classes']), \
        f"Expected output shape ({batch_size}, {config_model['num_classes']}), got {output_relu_v2.shape}"
    print("CNN2D-Only Model (ReLU) with different 2D width test passed.") 