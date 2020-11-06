"""gradient_xray_tuberculosis_frontal dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets.public_api as tfds
import io
import os
import pandas as pd
import numpy as np


# TODO(gradient_xray_tuberculosis_frontal): BibTeX citation
_CITATION = """
"""

# TODO(gradient_xray_tuberculosis_frontal):
_DESCRIPTION = """
"""

train_csv_path = '../train.csv'
val_csv_path = '../val.csv'
test_csv_path = '../test.csv'

class GradientXrayTuberculosisFrontal(tfds.core.GeneratorBasedBuilder):
  """TODO(gradient_xray_tuberculosis_frontal): Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
          # These are the features of your dataset like images, labels ...
          "image/name":
          tfds.features.Text(),
          "image":
          tfds.features.Image(shape=(None, None, 1),
                              dtype=tf.uint16,
                              encoding_format='png'),
          "label":
          tfds.features.ClassLabel(names=["negtive", "positive"]),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.

        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""


    train_df = pd.read_csv(train_csv_path, lineterminator='\n')
    val_df = pd.read_csv(val_csv_path, lineterminator='\n')
    test_df = pd.read_csv(test_csv_path, lineterminator='\n')

    
    return [
      tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN,
          gen_kwargs={
            'df':train_df,
          },
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.VALIDATION,
          gen_kwargs={
            'df':val_df,
          },
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.TEST,
          gen_kwargs={
            'df':test_df,
          },
      ),
    ]

  def _generate_examples(self, df):
    """Yields examples."""

    for idx, row in df.iterrows():    
#       path =tf.constant('oss://ai-dicom\x01id={access_key}\x02key={secret_key}\x02host=oss-cn-hangzhou-internal.aliyuncs.com/{path}'.format(
#                   bucket='ai-dicom',
#                   path='data/encrypted_dicoms/' + row.path,
#                   access_key=os.environ.get("OSS_ACCESS_KEY"),
#                   secret_key=os.environ.get("OSS_SECRET_KEY")
#               ))

#       image_bytes = gh_crypt.decrypt_bytes(
#         contents=tf.io.read_file(path),
#         chunk_size=64*1024, 
#         key_hex=os.environ['DECRYPT_KEY'])
      
#       image_tensor = tfio.image.decode_dicom_image(image_bytes)
#       image_tensor = tf.cast(tf.round(tf.image.resize_with_pad(image_tensor[0], 512, 512)), tf.uint16)

      record = {
        "image/name": row.path,
        "image": image_tensor.numpy(),
        "label": 'negtive' if row.label == 0 else 'positive'
      }
      yield idx, record
