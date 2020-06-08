"""mimic_cxr dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.image_classification import mimic_cxr


class MimicCxrTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = mimic_cxr.MimicCxr
  SPLITS = {
      "train": 4,  # Number of fake train example
      "validation": 1,
      "test": 2,  # Number of fake test example
  }


if __name__ == "__main__":
  tfds.testing.test_main()

