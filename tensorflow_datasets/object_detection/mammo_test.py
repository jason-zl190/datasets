"""mammo dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.object_detection import mammo


class MammoTest(tfds.testing.DatasetBuilderTestCase):
  # TODO(mammo):
  DATASET_CLASS = mammo.Mammo
  SPLITS = {
      "train": 3,  # Number of fake train example
      "test": 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
  tfds.testing.test_main()

