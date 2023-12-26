from easyocr import recognition
import os
import torch
import torchvision.transforms as transforms

recog_network = 'generation2'

network_params = {
    'input_channel': 1,
    'output_channel': 256,
    'hidden_size': 256
}

# see https://github.com/JaidedAI/EasyOCR/blob/ca9f9b0ac081f2874a603a5614ddaf9de40ac339/easyocr/config.py for other language config examples
character = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
symbol = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €"
model_path = "english_g2.pth"
separator_list = {}
lang_list = ['en']
package_dir = os.path.dirname(recognition.__file__)

dict_list = {}
for lang in lang_list:
    dict_list[lang] = os.path.join(package_dir, 'dict', lang + ".txt")

model, converter = recognition.get_recognizer(recog_network=recog_network,
                                              network_params=network_params,
                                              character=character,
                                              separator_list=separator_list,
                                              dict_list=dict_list,
                                              model_path=model_path,
                                              device='cpu',
                                              quantize=False)


# Define the dimensions of the input image
batch_size = 1
num_channels = 3
image_height = imgH = 64
image_width = 128
device = 'cpu'

# Create dummy input tensors for the image and text inputs
dummy_input = torch.randn(batch_size, num_channels, image_height, image_width)

# Define the maximum length of the text input
max_text_length = 10

dummy_text_input = torch.LongTensor(max_text_length, batch_size).random_(0, 10)

# Convert the input image to grayscale
grayscale_transform = transforms.Grayscale(num_output_channels=1)
grayscale_input = grayscale_transform(dummy_input)
grayscale_input = grayscale_transform(dummy_input.unsqueeze(0)).squeeze(0)

input_names = ["image_input", "text_input"]
output_names = ["output"]
dynamic_axes = {"image_input": {0: "batch_size"}, "text_input": {1: "batch_size"}}
opset_version = 12

torch.onnx.export(model=model,
                  args=(grayscale_input, dummy_text_input),
                  f="recog.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  #dynamic_axes=dynamic_axes,
                  opset_version=opset_version)