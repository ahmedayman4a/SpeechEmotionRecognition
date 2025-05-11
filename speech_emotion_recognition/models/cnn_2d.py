import torch
import torch.nn as nn

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 activation_module: nn.Module = None, dropout_rate=0.0):
        super(ResidualBlock2D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)
            
        padding = kernel_size // 2 # Same padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = activation_module
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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

class CNN2D(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 # img_height and img_width are now dynamic for input, not fixed resize targets
                 # num_mels_input can be used if needed, e.g., from config.N_MELS_IMG
                 dropout_rate=0.3, 
                 activation_module: nn.Module = None,
                 initial_out_channels=32, # Default, can be overridden from config. Paper: 32
                 block_channels=[32, 64, 128, 256], # Default, can be overridden from config. Paper: 32, 64, 512, 256
                 output_feature_size=512 # Target size after GAP for combined model
                ):
        super(CNN2D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)
        
        # Input shape: (batch_size, input_channels, H, W_variable) e.g. (N, 1, 64, 188)
        
        self.stem = nn.Sequential(
            # Initial conv, similar to ResNet stem, but adapted for spectrograms
            # Paper had 4 Conv Blocks. First one 32 filters, kernel 3x3
            nn.Conv2d(input_channels, initial_out_channels, kernel_size=7, stride=2, padding=3, bias=False), # H,W -> H/2, W/2
            nn.BatchNorm2d(initial_out_channels),
            activation_module,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # H,W -> H/4, W/4
        )
        
        current_channels = initial_out_channels
        self.res_blocks = nn.ModuleList()
        
        # Stacking residual blocks, adjusting channels and using MaxPool to reduce spatial dimensions
        # Example: Match paper's 4 stages of conv blocks with residual blocks.

        # Block 1 (corresponds to paper's 1st conv block: 32 filters)
        # Stem already produced initial_out_channels (e.g. 32), MaxPool already applied
        # So, first ResBlock operates on H/4, W/4
        self.res_blocks.append(ResidualBlock2D(current_channels, block_channels[0], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[0]
        # Paper: MaxPool after each conv block. Let's keep MaxPool between groups of ResBlocks.
        self.res_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)) # H,W -> H/8, W/8
        
        # Block 2 (corresponds to paper's 2nd conv block: 64 filters)
        self.res_blocks.append(ResidualBlock2D(current_channels, block_channels[1], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[1]
        self.res_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)) # H,W -> H/16, W/16

        # Block 3 (corresponds to paper's 3rd conv block: 512 filters, using 128 here)
        self.res_blocks.append(ResidualBlock2D(current_channels, block_channels[2], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[2]
        self.res_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)) # H,W -> H/32, W/32

        # Block 4 (corresponds to paper's 4th conv block: 256 filters)
        self.res_blocks.append(ResidualBlock2D(current_channels, block_channels[3], dropout_rate=dropout_rate, activation_module=activation_module))
        current_channels = block_channels[3]
        # No MaxPool after the last block before GAP
        
        # Global Average Pooling to handle variable width and produce fixed-size output
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final linear layer to project to the desired output_feature_size
        # The number of channels from the last res block (current_channels) is the input to this linear layer
        self.final_fc = nn.Linear(current_channels, output_feature_size)


        # print(f"CNN2D output before final_fc will have {current_channels} channels after GAP.")
        # print(f"CNN2D final output size: {output_feature_size}")

    def forward(self, x):
        # x input shape: (batch_size, input_channels, H_in, W_in)
        x = self.stem(x)
        for block_or_pool in self.res_blocks:
            x = block_or_pool(x)
        
        x = self.global_avg_pool(x) # Output: (N, current_channels, 1, 1)
        x = torch.flatten(x, 1)     # Output: (N, current_channels)
        if self.final_fc is not None:
            x = self.final_fc(x)        # Output: (N, output_feature_size)
        return x

if __name__ == '__main__':
    TARGET_OUTPUT_FEATURES_2D = 512 # Example target size
    INITIAL_CHANNELS_2D = 16 # Example for testing
    BLOCK_CHANNELS_2D = [16, 32, 64, 128] # Example for testing

    # Test with ReLU (default)
    print(f"Testing CNN2D (ReLU) with initial_channels={INITIAL_CHANNELS_2D}, block_channels={BLOCK_CHANNELS_2D}")
    model2d_relu = CNN2D(
        input_channels=1,
        activation_module=nn.ReLU(inplace=True),
        initial_out_channels=INITIAL_CHANNELS_2D,
        block_channels=BLOCK_CHANNELS_2D,
        output_feature_size=TARGET_OUTPUT_FEATURES_2D
    )
    dummy_input_2d_var_w = torch.randn(4, 1, 64, 188) # Example: H=64, W=188 (variable)
    output_2d_relu = model2d_relu(dummy_input_2d_var_w)
    print(f"CNN2D (ReLU ResNet) input shape: {dummy_input_2d_var_w.shape}")
    print(f"CNN2D (ReLU ResNet) output shape: {output_2d_relu.shape}")
    assert output_2d_relu.shape == (4, TARGET_OUTPUT_FEATURES_2D), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES_2D}), got {output_2d_relu.shape}"
    print("CNN2D (ReLU ResNet) with variable width input test passed.")

    # Test with SiLU
    print(f"\nTesting CNN2D (SiLU) with initial_channels={INITIAL_CHANNELS_2D}, block_channels={BLOCK_CHANNELS_2D}")
    model2d_silu = CNN2D(
        input_channels=1, 
        activation_module=nn.SiLU(),
        initial_out_channels=INITIAL_CHANNELS_2D,
        block_channels=BLOCK_CHANNELS_2D,
        output_feature_size=TARGET_OUTPUT_FEATURES_2D
    )
    output_2d_silu = model2d_silu(dummy_input_2d_var_w)
    print(f"CNN2D (SiLU ResNet) input shape: {dummy_input_2d_var_w.shape}")
    print(f"CNN2D (SiLU ResNet) output shape: {output_2d_silu.shape}")
    assert output_2d_silu.shape == (4, TARGET_OUTPUT_FEATURES_2D), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES_2D}), got {output_2d_silu.shape}"
    print("CNN2D (SiLU ResNet) with variable width input test passed.")

    # Test with a different width to ensure AdaptiveAvgPool2d works
    dummy_input_2d_other_w = torch.randn(4, 1, 64, 120) 
    output_2d_other_w = model2d_relu(dummy_input_2d_other_w)
    print(f"CNN2D (ReLU ResNet) input shape (other W): {dummy_input_2d_other_w.shape}")
    print(f"CNN2D (ReLU ResNet) output shape (other W): {output_2d_other_w.shape}")
    assert output_2d_other_w.shape == (4, TARGET_OUTPUT_FEATURES_2D), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES_2D}) for different width, got {output_2d_other_w.shape}"
    print("CNN2D (ReLU ResNet) with different variable width input test passed.") 