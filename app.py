# Lint as: python3
# 
# # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

# For image
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import collections
import detect



def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).
  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}



# Load the tflite interpreter
interpreter = tf.lite.Interpreter(
    model_path="coco_ssd_mobilenet/detect.tflite")
interpreter.allocate_tensors()

# Get input output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# pii_image =Image.open('opencv_frame_37.png')
# pii_image = ImageOps.fit(pii_image, (1286, 856), Image.ANTIALIAS)
# pii_image = pii_image.convert('RGB')
input_shape = input_details[0]['shape']
print(input_shape)


img = tf.io.read_file('opencv_frame_36.png')
img = tf.image.decode_png(img, channels=3)
img = tf.image.resize(img, (300, 300), antialias=True)


converted_image = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
print(type(converted_image))
print(converted_image.shape)


interpreter.set_tensor(input_details[0]['index'], converted_image)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.

labels = load_labels('coco_ssd_mobilenet/labelmap.txt')

objs = detect.get_output(interpreter, 0.4)

print('-------RESULTS--------')
if not objs:
    print('No objects detected')

for obj in objs:
    print(labels.get(obj.id, obj.id))
    print('  id:    ', obj.id)
    print('  score: ', obj.score)
    print('  bbox:  ', obj.bbox)

# if args.output:
#     image = image.convert('RGB')
#     draw_objects(ImageDraw.Draw(image), objs, labels)
#     image.save(args.output)
#     image.show()
