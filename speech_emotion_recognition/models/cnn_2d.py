import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 img_height=64, # Expected input image height
                 img_width=64,  # Expected input image width
                 dropout_rate=0.3, 
                 activation_module: nn.Module = None):
        super(CNN2D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)

        # Input shape: (batch_size, input_channels, img_height, img_width) e.g. (N, 1, 64, 64)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(64/2)=32
            nn.Dropout(dropout_rate)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(32/2)=16
            nn.Dropout(dropout_rate)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(16/2)=8
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            activation_module,
            nn.MaxPool2d(kernel_size=(2,2), stride=2, ceil_mode=True), # H,W_out = ceil(8/2)=4
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        # Expected flattened output: 256 (channels) * 4 (H) * 4 (W) = 4096

        # Helper to verify output size during init if needed
        # self._feature_size = self._get_conv_output_size(input_channels, img_height, img_width)
        # if self._feature_size != 4096:
        #     print(f"Warning: CNN2D calculated feature size is {self._feature_size}, expected 4096.")

    def _get_conv_output_size(self, input_channels, img_h, img_w):
        # For verification
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_h, img_w)
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, x):
        # x input shape: (N, C, H, W) -> (N, 1, 64, 64)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x) # Output: (N, 4096)
        return x

if __name__ == '__main__':
    # Test with ReLU (default, as per paper summary)
    model2d_relu = CNN2D(input_channels=1, img_height=64, img_width=64)
    dummy_input_2d = torch.randn(4, 1, 64, 64)
    output_2d_relu = model2d_relu(dummy_input_2d)
    print(f"CNN2D (ReLU) input shape: {dummy_input_2d.shape}")
    print(f"CNN2D (ReLU) output shape: {output_2d_relu.shape}")
    assert output_2d_relu.shape == (4, 4096), f"Expected output shape (4, 4096), got {output_2d_relu.shape}"
    print("CNN2D (ReLU) test passed.")

    # Test with SiLU (as per user's initial preference)
    model2d_silu = CNN2D(input_channels=1, img_height=64, img_width=64, activation_module=nn.SiLU())
    output_2d_silu = model2d_silu(dummy_input_2d)
    print(f"CNN2D (SiLU) input shape: {dummy_input_2d.shape}")
    print(f"CNN2D (SiLU) output shape: {output_2d_silu.shape}")
    assert output_2d_silu.shape == (4, 4096), f"Expected output shape (4, 4096), got {output_2d_silu.shape}"
    print("CNN2D (SiLU) test passed.")
    # Verify actual output size from an instance
    # print(f"Calculated output size for ReLU model: {model2d_relu._get_conv_output_size(1, 64, 64)}")
    # print(f"Calculated output size for SiLU model: {model2d_silu._get_conv_output_size(1, 64, 64)}") 