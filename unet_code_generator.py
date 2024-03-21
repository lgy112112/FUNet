import torch.nn as nn


def unet_code_generator(depth, n_channels, n_classes, bilinear=False):
    """
    Generates the Python code text for a UNet architecture with specified depth.

    Args:
        depth (int): The desired depth of the UNet model.
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
        bilinear (bool): Whether to use bilinear upsampling (default: False).

    Returns:
        str: The generated Python code text defining the UNet class.
    """

    code_text = ""

    # --- Beginning (remains the same) ---
    code_text += "from FUNet.unet_parts import *\n\n"
    code_text += "class UNet(nn.Module):\n"
    code_text += f"    def __init__(self, n_channels={n_channels}, n_classes={n_classes}, bilinear={bilinear}):\n"
    code_text += "        super(UNet, self).__init__()\n"
    code_text += "        self.n_channels = n_channels\n"
    code_text += "        self.n_classes = n_classes\n"
    code_text += "        self.bilinear = bilinear\n"
    code_text += "        factor = 2 if bilinear else 1\n\n"

    # --- Encoder blocks ---
    code_text += "        self.inc = DoubleConv(n_channels, 64)\n"  # Initial convolution
    for i in range(1, depth):
        in_channels = 64 * 2 ** (i - 1)
        out_channels = in_channels * 2
        divide = " // factor" if i == depth - 1 else ""  # Apply division only on last 'down'
        code_text += f"        self.down{i} = Down({in_channels}, {out_channels}{divide})\n"
    code_text += "\n"
    record = out_channels
    # --- Decoder blocks ---
    for i in range(depth - 1, 0, -1):
        in_channels = out_channels  # Corrected calculation
        out_channels = in_channels // 2
        divide = " // factor" if i != 1 else ""
        code_text += f"        self.up{i} = Up({in_channels}, {out_channels}{divide}, bilinear)\n"
    code_text += "        self.outc = OutConv(64, n_classes)\n\n"

    # --- Forward function ---
    code_text += "    def forward(self, x):\n"
    code_text += "        x1 = self.inc(x)\n"

    for i in range(1, depth):
        code_text += f"        x{i + 1} = self.down{i}(x{i})\n"

    # Corrected order for upsampling
    for i in range(depth - 1, 0, -1):  # (4,0,-1)
        if i == depth - 1:
            code_text += f"        x = self.up{i}(x{depth}, x{depth - 1})\n"
        else:
            code_text += f"        x = self.up{i}(x, x{i})\n"

    code_text += "        logits = self.outc(x)\n"
    code_text += "        return logits\n"

    return code_text
