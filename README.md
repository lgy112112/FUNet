# FUNet-Fast-UNet-Code-Generator

A flexible and user-friendly tool for generating custom PyTorch UNet implementations.

**Key Features**

* **Customization:** Easily define the depth, input channels, output classes, and upsampling methods (bilinear or transposed convolution) for your UNet architecture.
* **Speed and Efficiency:** Quickly generate the necessary Python code for your UNet model.
* **PyTorch Integration:** The generated code is fully compatible with PyTorch, allowing for seamless training and deployment.

**Getting Started**

1. **Installation:**
    ```bash
    git clone [https://github.com/lgy112112/FUNet-Fast-UNet-Code-Generator.git](https://github.com/lgy112112/FUNet.git)
    ```
    or, in notebook:
    ```bash
    !git clone [https://github.com/lgy112112/FUNet-Fast-UNet-Code-Generator.git](https://github.com/lgy112112/FUNet.git)
    ```

3. **Generate Your UNet Code:**
    ```python
    from FUNet.unet_code_generator import *

    # Example usage 
    my_unet_code = unet_code_generator(depth=4, n_channels=3, n_classes=2, bilinear=True)
    print(my_unet_code)
    ```

4. **Utilize the Code:**
    * Copy and paste the generated code into a Python file.
    * Create a PyTorch model instance using the generated class.
    * Train and use your UNet model as needed.

**Example**

```python
# ... (Import statements)

# Generate UNet code with depth 5, 1 input channel, 10 output classes, and bilinear upsampling
code_text = unet_code_generator(depth=5, n_channels=1, n_classes=10, bilinear=True)

# Define your UNet model using the generated code  
# Paste here

model = UNet()  # Create an instance of your generated UNet class

# ... (Training and usage of your model)
```

**Contributing**

We welcome contributions to improve this generator! 
* Fork the repository
* Create your branch
* Make changes and submit a pull request

**License**

This project is licensed under the MIT License: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) - feel free to use and distribute.
