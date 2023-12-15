# ./mnist/handler.py
from ts.torch_handler.base_handler import BaseHandler
from io import BytesIO
from PIL import Image
import base64
import numpy
import torch
import abc


class FooServe(BaseHandler, abc.ABC):
  def preprocess(self, data):
    base_img = data[0].get("body").get("img")
    base_img = base_img.replace("data:image/png;base64,", "")
    base_img = base64.b64decode(base_img)
    numpy_arr = numpy.frombuffer(base_img, numpy.uint8)
    pil_img = Image.open(BytesIO(numpy_arr)).convert('L')
    resized_image = pil_img.resize((28, 28), Image.BILINEAR)
    np_img = numpy.array(resized_image)
    ts_img = torch.tensor(np_img, dtype=torch.float).view(1, 1, 28, 28)
    ts_img /= 255.0
    return ts_img

  def inference(self, data, *args, **kwargs):
    with torch.no_grad():
      device_data = data.to(self.device)
      results = self.model(device_data, *args, **kwargs)
    return results

  def postprocess(self, data):
    output = torch.nn.functional.softmax(data, dim=1)
    return output.tolist()