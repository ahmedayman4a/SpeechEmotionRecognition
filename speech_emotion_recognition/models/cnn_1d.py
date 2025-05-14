import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, 
                 input_channels=1,
                 dropout_rate=0.3, 
                 activation_module: nn.Module = None):
        super(CNN1D, self).__init__()

        if activation_module is None:
            activation_module = nn.ReLU(inplace=True)
        
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 128),
            activation_module,
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), # L_out = ceil(162/2) = 81
            nn.Dropout(dropout_rate)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 256),
            activation_module,
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), # L_out = ceil(81/2) = 41
            nn.Dropout(dropout_rate)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 512),
            activation_module,
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), # L_out = ceil(41/2) = 21
            nn.Dropout(dropout_rate)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def _get_conv_output_size(self, input_channels, num_features_input_dim):
        # For verification
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, num_features_input_dim)
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.global_avg_pool(x)
            x = torch.flatten(x, 1) # Flatten after GAP to get (N, C)
            return x.shape[1]

    def forward(self, x):
        # x input shape: (batch_size, input_channels, num_features_input_dim) -> (N, 1, 162)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x) # Output: (N, 512, 1)
        x = torch.flatten(x, 1)   # Output: (N, 512)
        return x

if __name__ == '__main__':
    # Test with ReLU (default, as per paper summary)
    model1d_relu = CNN1D(input_channels=1, num_features_input_dim=162)
    dummy_input_1d = torch.randn(4, 1, 162) 
    output_1d_relu = model1d_relu(dummy_input_1d)
    print(f"CNN1D (ReLU) input shape: {dummy_input_1d.shape}")
    print(f"CNN1D (ReLU) output shape: {output_1d_relu.shape}") 
    assert output_1d_relu.shape == (4, 512), f"Expected output shape (4, 512), got {output_1d_relu.shape}" # Updated assertion
    print("CNN1D (ReLU) test passed.")

    # Test with SiLU (as per user's initial preference)
    model1d_silu = CNN1D(input_channels=1, num_features_input_dim=162, activation_module=nn.SiLU())
    output_1d_silu = model1d_silu(dummy_input_1d)
    print(f"CNN1D (SiLU) input shape: {dummy_input_1d.shape}")
    print(f"CNN1D (SiLU) output shape: {output_1d_silu.shape}")
    assert output_1d_silu.shape == (4, 512), f"Expected output shape (4, 512), got {output_1d_silu.shape}" # Updated assertion
    print("CNN1D (SiLU) test passed.")
    # Verify actual output size from an instance
    # print(f"Calculated output size for ReLU model: {model1d_relu._get_conv_output_size(1, 162)}")
    # print(f"Calculated output size for SiLU model: {model1d_silu._get_conv_output_size(1, 162)}") 