import tensorflow as tf
import argparse
import json
import sys
import os
from PIL import Image

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('input_json', '', 'Path to feed.json')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_integer('limit', None, 'Number of dataset instances to create')

FLAGS = flags.FLAGS


def read_image(filename):
    image = Image.open(filename)
    width, height = image.size


    # # Don't use tf.image.decode_image, or the output shape will be undefined
    # image = tf.image.decode_jpeg(image_string, channels=3)
    # # This will convert to float values in [0, 1]
    # image = tf.image.convert_image_dtype(image, tf.float32)

    # resized_image = tf.image.resize_images(image, [height, width])
    with tf.io.gfile.GFile (filename, 'rb') as fid:
        encoded_jpg = fid.read()

    return encoded_jpg, height, width


def create_tf_example(dataset_dir, obj):
    image_path = obj['image']

    # TODO(user): Populate the following variables from your example.
    full_image_path = os.path.join(dataset_dir, image_path)  # Filename of the image. Empty if image is not from file
    encoded_image_data,height, width = read_image(full_image_path)  # Encoded image bytes
    filename = os.path.basename(full_image_path).encode()
    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = [1.0]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [1.0]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [1.0]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [1.0]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = ['BMW'.encode()]  # List of string class name of bounding box (1 per box)
    classes = [1]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    limit = FLAGS.limit
    # TODO(user): Write code to read in your dataset to examples variable

    json_file = FLAGS.input_json
    dataset_dir = os.path.dirname(json_file)
    with open(json_file) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            tf_example = create_tf_example(dataset_dir, obj)
            writer.write(tf_example.SerializeToString())

            if limit is not None and i > limit:
                break

    writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run()
