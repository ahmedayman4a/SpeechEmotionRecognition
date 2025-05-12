import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 activation_module: nn.Module = None, dropout_rate=0.0):
        super(ResidualBlock1D, self).__init__()
        
        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)
            
        padding = kernel_size // 2 # Same padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = activation_module
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.final_act = activation_module

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.final_act(out)
        return out

class CNN1D(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 num_features_input_dim=162, # Length of the input 1D sequence
                 dropout_rate=0.3, 
                 activation_module: nn.Module = None,
                 initial_out_channels=64, # Default, can be overridden from config
                 block_channels=[64, 128, 256, 512], # Default, can be overridden from config
                 output_feature_size=256 # Target size after GAP for combined model
                 ):
        super(CNN1D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, initial_out_channels, kernel_size=7, stride=2, padding=3, bias=False), # L_out = ceil(162/2) = 81
            nn.BatchNorm1d(initial_out_channels),
            activation_module,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # L_out = ceil(81/2) = 41
        )
        
        current_channels = initial_out_channels
        self.res_blocks = nn.ModuleList()
        
        # Example: Four residual blocks with increasing channels and some max pooling
        # Adjust strides and pooling to manage sequence length reduction
        
        # Block 1
        self.res_blocks.append(ResidualBlock1D(current_channels, block_channels[0], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[0]
        self.res_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)) # L_out = ceil(41/2) = 21
        
        # Block 2
        self.res_blocks.append(ResidualBlock1D(current_channels, block_channels[1], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[1]
        self.res_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)) # L_out = ceil(21/2) = 11
        
        # Block 3
        self.res_blocks.append(ResidualBlock1D(current_channels, block_channels[2], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[2]
        

        # Global Average Pooling to get fixed size output
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final linear layer to project to the desired output_feature_size
        # The number of channels from the last res block (current_channels) is the input to this linear layer
        self.final_fc = nn.Linear(current_channels, output_feature_size)
        
        # For verification of output size calculation during development
        # self._calculated_feature_size = self._get_conv_output_size(input_channels, num_features_input_dim)
        # print(f"CNN1D output before final_fc will have {current_channels} channels after GAP.")
        # print(f"CNN1D final output size: {output_feature_size}")


    def _get_conv_output_size(self, input_channels, num_features_input_dim):
        # For verification
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, num_features_input_dim)
            x = self.stem(dummy_input)
            for block_or_pool in self.res_blocks:
                x = block_or_pool(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1)
            # Before final_fc
            return x.shape[1] 

    def forward(self, x):
        # x input shape: (batch_size, input_channels, num_features_input_dim) -> (N, 1, 162)
        x = self.stem(x)
        for block_or_pool in self.res_blocks:
            x = block_or_pool(x)
        
        x = self.global_avg_pool(x) # Output: (N, current_channels, 1)
        x = torch.flatten(x, 1)     # Output: (N, current_channels)
        x = self.final_fc(x)        # Output: (N, output_feature_size)
        return x

if __name__ == '__main__':
    TARGET_OUTPUT_FEATURES = 256 # Example target size
    INITIAL_CHANNELS_1D = 32 # Example different initial channels for testing
    BLOCK_CHANNELS_1D = [32, 64, 128, 256] # Example different block channels for testing

    # Test with ReLU 
    print(f"Testing CNN1D with initial_channels={INITIAL_CHANNELS_1D}, block_channels={BLOCK_CHANNELS_1D}")
    model1d_relu = CNN1D(
        input_channels=1, 
        num_features_input_dim=162, 
        activation_module=nn.ReLU(inplace=True),
        initial_out_channels=INITIAL_CHANNELS_1D,
        block_channels=BLOCK_CHANNELS_1D,
        output_feature_size=TARGET_OUTPUT_FEATURES
    )
    dummy_input_1d = torch.randn(4, 1, 162) 
    output_1d_relu = model1d_relu(dummy_input_1d)
    print(f"CNN1D (ReLU ResNet) input shape: {dummy_input_1d.shape}")
    print(f"CNN1D (ReLU ResNet) output shape: {output_1d_relu.shape}") 
    assert output_1d_relu.shape == (4, TARGET_OUTPUT_FEATURES), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES}), got {output_1d_relu.shape}"
    print("CNN1D (ReLU ResNet) test passed.")

    # Test with SiLU
    print(f"\nTesting CNN1D (SiLU) with initial_channels={INITIAL_CHANNELS_1D}, block_channels={BLOCK_CHANNELS_1D}")
    model1d_silu = CNN1D(
        input_channels=1, 
        num_features_input_dim=162, 
        activation_module=nn.SiLU(),
        initial_out_channels=INITIAL_CHANNELS_1D,
        block_channels=BLOCK_CHANNELS_1D,
        output_feature_size=TARGET_OUTPUT_FEATURES
    )
    output_1d_silu = model1d_silu(dummy_input_1d)
    print(f"CNN1D (SiLU ResNet) input shape: {dummy_input_1d.shape}")
    print(f"CNN1D (SiLU ResNet) output shape: {output_1d_silu.shape}")
    assert output_1d_silu.shape == (4, TARGET_OUTPUT_FEATURES), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES}), got {output_1d_silu.shape}"
    print("CNN1D (SiLU ResNet) test passed.")
    
    # print(f"Calculated feature size for ReLU model (before final_fc): {model1d_relu._get_conv_output_size(1, 162)}")
    # print(f"Final output size for ReLU model (after final_fc): {model1d_relu(torch.randn(1,1,162)).shape[1]}") 