from easyocr.recognition import get_text

from easyocr import recognition
import os
import torch
import torchvision.transforms as transforms
from easyocr.utils import reformat_input, get_image_list

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
lang = 'en'
package_dir = os.path.dirname(recognition.__file__)

dict_list = {}

dict_list[lang] = os.path.join(package_dir, 'dict', lang + ".txt")

model, converter = recognition.get_recognizer(recog_network=recog_network,
                                              network_params=network_params,
                                              character=character,
                                              separator_list=separator_list,
                                              dict_list=dict_list,
                                              model_path=model_path,
                                              device='cpu',
                                              quantize=False)

img, img_cv_grey = reformat_input('/content/outputs/image_crops/crop_0.png')

y_max, x_max = img_cv_grey.shape
imgH = 64
horizontal_list = [[0, x_max, 0, y_max]]

lang_char = []
char_file = os.path.join(package_dir, 'character', lang + "_char.txt")
with open(char_file, "r", encoding="utf-8-sig") as input_file:
    char_list = input_file.read().splitlines()
lang_char += char_list
lang_char = set(lang_char).union(set(symbol))

ignore_char = ''.join(set(character) - set(lang_char))

result = []

for bbox in horizontal_list:
    h_list = [bbox]
    f_list = []
    image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=64)  # 64 is default value
    result0 = get_text(character, imgH, int(max_width), converter, image_list,
                       ignore_char, 'greedy',
                       beamWidth=5,
                       batch_size=1,
                       contrast_ths=0.1,
                       adjust_contrast=0.5,
                       filter_ths=0.003,
                       workers=0,
                       device='cpu')
    result += result0
