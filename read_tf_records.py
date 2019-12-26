import tensorflow as tf
import IPython.display as display
tf.enable_eager_execution()

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.int64),
    'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.int64),
    'image/object/class/text': tf.io.FixedLenFeature([], tf.int64),
    'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

print(parsed_image_dataset)

for image_features in parsed_image_dataset:
  image_raw = image_features['image/encoded'].numpy()
  display.display(display.Image(data=image_raw))
