from easyocr import detection
import torch

# load model using CPU - default
model = detection.get_detector(trained_model='craft_mlt_25k.pth', device='cpu', quantize=False)

dummy_input = torch.randn(1, 3, 384, 512)
dynamic_axes = {
    'image': {
        2: 'width',
        3: 'height'}
}
torch.onnx.export(model=model,
                  args=dummy_input,
                  f="craft.onnx",
                  input_names=['image'],
                  output_names=['output'],
                  opset_version=11,
                  dynamic_axes=dynamic_axes
                  )
