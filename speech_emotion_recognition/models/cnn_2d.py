import torch
import torch.nn as nn

class Bottleneck2D(nn.Module):
    """Residual Bottleneck Block for 2D CNN"""
    expansion = 4

    def __init__(self, in_channels, bottleneck_channels, stride=1, downsample=None, activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(1, out_channels)
        
        self.activation = activation_fn
        self.downsample = downsample

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
        out = self.activation(out)

        return out


class CNN2D(nn.Module):
    def __init__(self, input_channels=1,
                 block=Bottleneck2D, layers=[2, 2, 2, 2], 
                 initial_out_channels=64, 
                 activation_name='relu', 
                 dropout_rate=0.3):
        super().__init__()
        
        if activation_name.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_name.lower() == 'silu' or activation_name.lower() == 'swish':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.in_channels = initial_out_channels
        
        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.GroupNorm(1, self.in_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Stages
        self.layer1 = self._make_layer(block, initial_out_channels, layers[0], activation_fn=self.activation)
        self.layer2 = self._make_layer(block, initial_out_channels * 2, layers[1], stride=2, activation_fn=self.activation)
        self.layer3 = self._make_layer(block, initial_out_channels * 4, layers[2], stride=2, activation_fn=self.activation)
        self.layer4 = self._make_layer(block, initial_out_channels * 8, layers[3], stride=2, activation_fn=self.activation)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self._output_dim = initial_out_channels * 8 * block.expansion

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, bottleneck_channels, num_blocks, stride=1, activation_fn=nn.ReLU(inplace=True)):
        downsample = None
        out_channels = bottleneck_channels * block.expansion
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, bottleneck_channels, stride, downsample, activation_fn=activation_fn))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, bottleneck_channels, activation_fn=activation_fn))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [N, C_in, H, W] e.g., [N, 1, 64, 64]
        x = self.conv1(x)    # [N, 64, H/2, W/2]
        x = self.norm1(x)
        x = self.activation(x)
        x = self.pool1(x)    # [N, 64, H/4, W/4]

        x = self.layer1(x)   # [N, 64*4, H/4, W/4]
        x = self.layer2(x)   # [N, 128*4, H/8, W/8]
        x = self.layer3(x)   # [N, 256*4, H/16, W/16]
        x = self.layer4(x)   # [N, 512*4, H/32, W/32]

        x = self.avgpool(x)  # [N, 512*4, 1, 1]
        x = self.flatten(x)  # [N, 512*4]
        x = self.dropout(x)

        return x

    @property
    def output_dim(self):
        return self._output_dim

if __name__ == '__main__':
    TARGET_OUTPUT_FEATURES_2D = 1024 # Example target size
    INITIAL_CHANNELS_2D = 32 # Example for testing
    BLOCK_CHANNELS_2D = [16, 32, 64, 128] # Example for testing

    # Test with ReLU (default)
    print(f"Testing CNN2D (ReLU) with initial_channels={INITIAL_CHANNELS_2D}, block_channels={BLOCK_CHANNELS_2D}")
    model2d_relu = CNN2D(
        input_channels=1,
        activation_name='relu',
        initial_out_channels=INITIAL_CHANNELS_2D,
        layers=BLOCK_CHANNELS_2D,
        dropout_rate=0.3
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
        activation_name='silu',
        initial_out_channels=INITIAL_CHANNELS_2D,
        layers=BLOCK_CHANNELS_2D,
        dropout_rate=0.3
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