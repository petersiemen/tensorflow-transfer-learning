import tensorflow as tf
import argparse
import json
import sys
import os
from PIL import Image
import math

from object_detection.utils import dataset_util


def read_image(filename):
    image = Image.open(filename)
    width, height = image.size

    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()

    return encoded_jpg, height, width


def create_label_map(labels_csv):
    with open(labels_csv) as f:
        makes = [line.strip() for line in f.readlines()]
    label_map = {}
    for i, make in enumerate(makes):
        label_map[make] = i
    return label_map


def write_label_map(label_map, dir):
    with open(os.path.join(dir, 'label_map.pbtxt'), 'w') as f:
        for text, label in label_map.items():
            f.write(
                """
item {
    id: %d
    name: '%s'
}
                """ % (label, text))


def create_tf_example(dataset_dir, obj, label_map):
    image_path = obj['image']

    # TODO(user): Populate the following variables from your example.
    full_image_path = os.path.join(dataset_dir, image_path)  # Filename of the image. Empty if image is not from file
    encoded_image_data, height, width = read_image(full_image_path)  # Encoded image bytes
    filename = os.path.basename(full_image_path).encode()
    image_format = b'jpeg'  # b'jpeg' or b'png'

    x = obj['bbox'][0]
    y = obj['bbox'][1]
    w = obj['bbox'][2]
    h = obj['bbox'][3]

    xmins = [x - w / 2]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [x + w / 2]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [y - h / 2]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [y + h / 2]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    make = obj['make']
    classes_text = [make.encode()]  # List of string class name of bounding box (1 per box)
    classes = [label_map[make]]  # List of integer class id of bounding box (1 per box)

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


class WriterWrapper:
    def __init__(self, images_per_file, total, out_dir):
        self.images_per_file = images_per_file
        self.number_of_partitions = math.ceil(total / images_per_file)
        self.n = 0
        self.idx_of_current_partition = 0
        self.out_dir = out_dir
        self.writer = tf.io.TFRecordWriter(
            os.path.join(self.out_dir, self._get_filename()))

    def _get_filename(self):
        return f'images.record-{self.idx_of_current_partition:05}-of-{self.number_of_partitions:05}'

    def _get_writer(self):
        if self.n >= self.images_per_file:
            self.writer.close()
            self.idx_of_current_partition += 1
            self.writer = tf.io.TFRecordWriter(os.path.join(self.out_dir, self._get_filename()))
            self.n = 0

        self.n += 1
        return self.writer

    def write(self, serialized):
        self._get_writer().write(serialized)

    def close(self):
        self.writer.close()


def run():
    parser = argparse.ArgumentParser('convert_to_tf_records.py')

    parser.add_argument("--annotations-file", dest="annotations_file",
                        help="annotations file", metavar="FILE")

    parser.add_argument("--batch-size", dest="batch_size",
                        type=int,
                        default=250,
                        help="batch_size for number of images per record file (default: 250)")

    parser.add_argument("--out-dir", dest="out_dir",
                        help="where to write to")

    parser.add_argument("--labels-file", dest="labels_file",
                        help="where to write to")

    parser.add_argument("--limit", dest="limit",
                        type=int,
                        default=None,
                        help="maxium number of images to process (default: None)")

    args = parser.parse_args()
    if args.annotations_file is None or args.out_dir is None or args.labels_file is None:
        parser.print_help()
        sys.exit(1)

    json_file = args.annotations_file
    out_dir = args.out_dir
    batch_size = args.batch_size
    labels_file = args.labels_file

    with open(json_file) as f:
        total = len(f.readlines())
    limit = args.limit if args.limit is not None else total

    writer_wrapper = WriterWrapper(batch_size, limit, out_dir)

    label_map = create_label_map(labels_file)
    write_label_map(label_map, out_dir)

    dataset_dir = os.path.dirname(json_file)
    with open(json_file) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            tf_example = create_tf_example(dataset_dir, obj, label_map)
            writer_wrapper.write(tf_example.SerializeToString())

            if limit is not None and i > limit:
                break

    writer_wrapper.close()


if __name__ == '__main__':
    run()
