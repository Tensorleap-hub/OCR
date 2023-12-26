import onnxruntime as rt
import cv2
import numpy as np
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.general_utils import draw_bounding_boxes
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from easyocr.utils import reformat_input, get_image_list, group_text_box, diff
import torch

# image preprocessing (input_encoder)
# Read input image
img, img_cv_grey = reformat_input('../examples/example3.png')
# Resize and normalize input image
img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, 512, interpolation=cv2.INTER_LINEAR, mag_ratio=1.)
ratio_h = ratio_w = 1 / target_ratio
x = normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

# inference
# Create ONNX Runtime session and load model
providers = ['CPUExecutionProvider']
session = rt.InferenceSession("craft.onnx", providers=providers)
input_name = session.get_inputs()[0].name

# Prepare input tensor for inference
inp = {input_name: x.numpy()}

# Run inference and get output
y, feature = session.run(None, inp)

# output decoding
# Extract score and link maps
boxes_list, polys_list = [], []
score_text = y[0, :, :, 0]
score_link = y[0, :, :, 1]

# Post-processing to obtain bounding boxes and polygons
boxes, polys, mapper = getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4, poly=True)
boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

boxes = list(boxes)
polys = list(polys)
for k in range(len(polys)):
    if polys[k] is None:
        polys[k] = boxes[k]
boxes_list.append(boxes)
polys_list.append(polys)

for polys in polys_list:
    single_img_result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        single_img_result.append(poly)

text_box_list = [single_img_result]
horizontal_list_agg, free_list_agg = [], []
min_size = 20
for text_box in text_box_list:
    horizontal_list, free_list = group_text_box(text_box, slope_ths=0.1,
                                                ycenter_ths=0.5, height_ths=0.5,
                                                width_ths=0.5, add_margin=0.1,
                                                sort_output=True)

    if min_size:
        horizontal_list = [i for i in horizontal_list if max(
            i[1] - i[0], i[3] - i[2]) > min_size]
        free_list = [i for i in free_list if max(
            diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
    horizontal_list_agg.append(horizontal_list)
    free_list_agg.append(free_list)
# image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
# draw_bounding_boxes(image=img, bbox_array=boxes)
horizontal_list, free_list = horizontal_list_agg[0], free_list_agg[0]
# for bbox in horizontal_list:
#     h_list = [bbox]
#     f_list = []
#     image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=64)
#     cv2.imshow('crop', image_list[0][1])
#     cv2.waitKey(0)
for bbox in free_list:
    h_list = []
    f_list = [bbox]
    image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height=64)
    cv2.imshow('crop', image_list[0][1])
    cv2.waitKey(0)
a = 0
