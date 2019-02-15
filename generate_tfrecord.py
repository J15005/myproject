"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append('C:/Users/J15005/Desktop/myproject/models/research')
# sys.path.append('C:/Desktop/myproject/models/research/object_detection/utils')

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV inpu')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'iwan':
        return 1
    elif row_label == 'ryanwan':
        return 2
    elif row_label == 'sanwan':
        return 3
    elif row_label == 'suwan':
        return 4
    elif row_label == 'uwan':
        return 5
    elif row_label == 'rowan':
        return 6
    elif row_label == 'tiwan':
        return 7
    elif row_label == 'pawan':
        return 8
    elif row_label == 'kyuwan':
        return 9
    elif row_label == 'ipin':
        return 10
    elif row_label == 'ryanpin':
        return 11
    elif row_label == 'sanpin':
        return 12
    elif row_label == 'supin':
        return 13
    elif row_label == 'upin':
        return 14
    elif row_label == 'ropin':
        return 15
    elif row_label == 'tipin':
        return 16
    elif row_label == 'papin':
        return 17
    elif row_label == 'kyupin':
        return 18
    elif row_label == 'isou':
        return 19
    elif row_label == 'ryansou':
        return 20
    elif row_label == 'sansou':
        return 21
    elif row_label == 'susou':
        return 22
    elif row_label == 'usou':
        return 23
    elif row_label == 'rosou':
        return 24
    elif row_label == 'tisou':
        return 25
    elif row_label == 'pasou':
        return 26
    elif row_label == 'kyusou':
        return 27
    elif row_label == 'ton':
        return 28
    elif row_label == 'nan':
        return 29
    elif row_label == 'sya':
        return 30
    elif row_label == 'pe':
        return 31
    elif row_label == 'haku':
        return 32
    elif row_label == 'hatu':
        return 33
    elif row_label == 'tyun':
        return 34
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        row['class'] = str(row['class'])# 追加 ######################################
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
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
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), 'images/JPEGImages')
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
