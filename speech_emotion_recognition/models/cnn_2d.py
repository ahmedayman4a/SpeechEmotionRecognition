import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 dropout_rate=0.3, 
                 activation_module: nn.Module = None):
        super(CNN2D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)

        # Input shape: (batch_size, input_channels, img_height, img_width) e.g. (N, 1, 64, 64)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.GroupNorm(1, 32),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(64/2)=32
            nn.Dropout(dropout_rate)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.GroupNorm(1, 64),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(32/2)=16
            nn.Dropout(dropout_rate)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.GroupNorm(1, 512),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(16/2)=8
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=1, padding=1),
            nn.GroupNorm(1, 1024),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(8/2)=4
            nn.Dropout(dropout_rate)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def _get_conv_output_size(self, input_channels, img_h, img_w):
        # For verification
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_h, img_w)
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1) # Flatten after GAP to get (N, C)
            return x.shape[1]

    def forward(self, x):
        # x input shape: (N, C, H, W) -> (N, 1, 64, 64)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x) # Output: (N, 1024, 1, 1)
        x = torch.flatten(x, 1)   # Output: (N, 1024)
        return x

if __name__ == '__main__':
    # Test with ReLU (default, as per paper summary)
    model2d_relu = CNN2D(input_channels=1)
    dummy_input_2d = torch.randn(4, 1, 64, 64)
    output_2d_relu = model2d_relu(dummy_input_2d)
    print(f"CNN2D (ReLU) input shape: {dummy_input_2d.shape}")
    print(f"CNN2D (ReLU) output shape: {output_2d_relu.shape}")
    assert output_2d_relu.shape == (4, 1024), f"Expected output shape (4, 1024), got {output_2d_relu.shape}" # Updated assertion
    print("CNN2D (ReLU) test passed.")

    # Test with SiLU (as per user's initial preference)
    model2d_silu = CNN2D(input_channels=1, activation_module=nn.SiLU())
    output_2d_silu = model2d_silu(dummy_input_2d)
    print(f"CNN2D (SiLU) input shape: {dummy_input_2d.shape}")
    print(f"CNN2D (SiLU) output shape: {output_2d_silu.shape}")
    assert output_2d_silu.shape == (4, 1024), f"Expected output shape (4, 1024), got {output_2d_silu.shape}" # Updated assertion
    print("CNN2D (SiLU) test passed.")
    # Verify actual output size from an instance
    # print(f"Calculated output size for ReLU model: {model2d_relu._get_conv_output_size(1, 64, 64)}")
    # print(f"Calculated output size for SiLU model: {model2d_silu._get_conv_output_size(1, 64, 64)}") 