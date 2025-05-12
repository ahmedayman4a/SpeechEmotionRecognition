import torch
import torch.nn as nn

class Bottleneck1D(nn.Module):
    """Residual Bottleneck Block for 1D CNN"""
    expansion = 4 # Bottleneck expands channels by 4

    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None, activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion
        
        # 1x1 conv
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        # Using GroupNorm(1, ...) as a LayerNorm alternative suitable for Conv layers
        self.norm1 = nn.GroupNorm(1, bottleneck_channels) 
        
        # 3x1 conv (with stride for downsampling)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, bottleneck_channels)
        
        # 1x1 conv
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(1, out_channels)
        
        self.activation = activation_fn
        self.downsample = downsample # For residual connection if shapes mismatch

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out) # Final activation after adding residual

        return out


class CNN1D(nn.Module):
    def __init__(self, input_channels=1, num_features_dim=162, # Keep input dims for clarity
                 block=Bottleneck1D, layers=[2, 2, 2, 2], # ResNet-18/34 like depth
                 initial_out_channels=64, # Base channel count
                 activation_name='relu', 
                 dropout_rate=0.3):
        super().__init__()
        
        if activation_name.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")
            
        self.in_channels = initial_out_channels # Track current channels for stages
        
        # Initial Convolutional Layer (ResNet-like)
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.GroupNorm(1, self.in_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Stages
        self.layer1 = self._make_layer(block, initial_out_channels, layers[0], activation_fn=self.activation)
        self.layer2 = self._make_layer(block, initial_out_channels * 2, layers[1], stride=2, activation_fn=self.activation)
        self.layer3 = self._make_layer(block, initial_out_channels * 4, layers[2], stride=2, activation_fn=self.activation)
        self.layer4 = self._make_layer(block, initial_out_channels * 8, layers[3], stride=2, activation_fn=self.activation)

        # Global Average Pooling and Output Dimension
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self._output_dim = initial_out_channels * 8 * block.expansion # Output dim is channels after last stage

        # Optional Dropout before final MLP head in CombinedModel
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Weight Initialization (optional but recommended)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, bottleneck_channels, num_blocks, stride=1, activation_fn=nn.ReLU(inplace=True)):
        downsample = None
        out_channels = bottleneck_channels * block.expansion
        
        # Downsample if stride > 1 or in_channels doesn't match out_channels
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, bottleneck_channels, stride, downsample, activation_fn=activation_fn))
        self.in_channels = out_channels # Update in_channels for subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, bottleneck_channels, activation_fn=activation_fn))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x initial shape: [N, C_in, L_in] = [N, 1, 162] for this project
        
        x = self.conv1(x)   # Example output: [N, 64, L_in/2]
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool1(x)   # Example output: [N, 64, L_in/4]

        x = self.layer1(x)  # Example output: [N, 64*4, L_in/4]
        x = self.layer2(x)  # Example output: [N, 128*4, L_in/8]
        x = self.layer3(x)  # Example output: [N, 256*4, L_in/16]
        x = self.layer4(x)  # Example output: [N, 512*4, L_in/32]

        x = self.avgpool(x) # Output: [N, 512*4, 1]
        x = self.flatten(x) # Output: [N, 512*4]
        x = self.dropout(x) # Apply dropout

        return x

    @property
    def output_dim(self):
        """Returns the output dimension of the flattened features."""
        return self._output_dim

if __name__ == '__main__':
    TARGET_OUTPUT_FEATURES = 1024 # Example target size
    INITIAL_CHANNELS_1D = 32 # Example different initial channels for testing
    BLOCK_CHANNELS_1D = [32, 64, 128, 256] # Example different block channels for testing

    # Test with ReLU 
    print(f"Testing CNN1D with initial_channels={INITIAL_CHANNELS_1D}, block_channels={BLOCK_CHANNELS_1D}")
    model1d_relu = CNN1D(
        input_channels=1, 
        num_features_dim=162, 
        activation_name='relu',
        initial_out_channels=INITIAL_CHANNELS_1D,
        layers=BLOCK_CHANNELS_1D,
        dropout_rate=0.3
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
        num_features_dim=162, 
        activation_name='silu',
        initial_out_channels=INITIAL_CHANNELS_1D,
        layers=BLOCK_CHANNELS_1D,
        dropout_rate=0.3
    )
    output_1d_silu = model1d_silu(dummy_input_1d)
    print(f"CNN1D (SiLU ResNet) input shape: {dummy_input_1d.shape}")
    print(f"CNN1D (SiLU ResNet) output shape: {output_1d_silu.shape}")
    assert output_1d_silu.shape == (4, TARGET_OUTPUT_FEATURES), f"Expected output shape (4, {TARGET_OUTPUT_FEATURES}), got {output_1d_silu.shape}"
    print("CNN1D (SiLU ResNet) test passed.")
    
    # print(f"Calculated feature size for ReLU model (before final_fc): {model1d_relu._get_conv_output_size(1, 162)}")
    # print(f"Final output size for ReLU model (after final_fc): {model1d_relu(torch.randn(1,1,162)).shape[1]}") 