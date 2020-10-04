"""mammo dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import utils
import tensorflow as tf
import os
import numpy as np
import ast

# TODO(mammo): BibTeX citation
_CITATION = """
"""

# TODO(mammo):
_DESCRIPTION = """
"""

#LESION_LABELS = ("mass_calc", 'calc', 'mass', 'ad', 'asym', 'lymph')
LESION_LABELS = ['mass']
LESION_TYPES = {
  '11000':'mass',#'mass_calc',
  '10000':'calc',
  '01000':'mass',
  '00100':'mass',#'ad',
  '00010':'mass',#'asym',
  '00001':'mass',#'lymph',
}

CLASS_LABELS = ("Benign", "Malignant")
RESULT_LABELS = ("TP", "FP")

CSV_PATH = os.path.join("..", "mass_calc_refined_positive.csv")

class Mammo(tfds.core.GeneratorBasedBuilder):
  """TODO(mammo): Short description of my dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You have to put the data in the manual_dir.
  """
  # TODO(mammo): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(mammo): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Image(shape=(None, None, 1),
                              dtype=tf.uint16,
                              encoding_format='png'),
            "rows": tfds.features.Tensor(shape=(), dtype=tf.int32),
            "cols": tfds.features.Tensor(shape=(), dtype=tf.int32),
            "window_center": tfds.features.Tensor(shape=(), dtype=tf.float32),
            "window_width": tfds.features.Tensor(shape=(), dtype=tf.float32),
            "image/filename": tfds.features.Text(),
            "objects": tfds.features.Sequence({
                "label": tfds.features.ClassLabel(names=LESION_LABELS),
                "bbox": tfds.features.BBoxFeature(),
            }),
            "labels": tfds.features.Sequence(
                tfds.features.ClassLabel(names=LESION_LABELS)),
            "class": tfds.features.ClassLabel(names=CLASS_LABELS),
            "result": tfds.features.ClassLabel(names=RESULT_LABELS),
        }),

        # Homepage of the dataset for documentation
        homepage='https://scholar.google.com/citations?user=zWViwVAAAAAJ&hl=en',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(mammo): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    data_path = dl_manager.manual_dir
    annParser = AnnParser(CSV_PATH)
    
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"data_path": data_path,
                        "split": annParser.ann['train'],},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={"data_path": data_path,
                        "split": annParser.ann['validation'],},
        ),
    ]

  def _generate_examples(self, data_path, split):
    """Yields examples."""
    # TODO(mammo): Yields (key, example) tuples from the dataset
    for idx, value in enumerate(split.values):
      # collect annotations
      fake_id, image_path, window_center, window_width, \
      rows, columns, manufacturer, \
      lesion_type, bounding_box, Split_Num, class_label, result_label = value

      def format_bbox(bbox, height, width):
        Ax, Ay, Bx, By = bbox
        ymin, xmin = min(Ay, By), min(Ax, Bx)
        ymax, xmax = max(Ay, By), max(Ax, Bx)
        return ymin / height, xmin / width, ymax / height, xmax / width 
      
      # if 'calc' in lesion_type or ('calc' == lesion_type):
      #   print("{},{}".format(image_path, lesion_type))
      objects = [{"label": x[0], 
                  "bbox": tfds.features.BBox(*format_bbox(x[1], int(rows), int(columns)))
                  } for x in zip(lesion_type, bounding_box)] 
      # build example
      record = {
          "image": os.path.join(data_path, image_path),
          "rows": int(rows),
          "cols": int(columns),
          "window_center": window_center,
          "window_width": window_width,
          "image/filename": image_path,
          "objects": objects,
          "labels": sorted(set(obj["label"] for obj in objects)),
          "class": class_label,
          "result": result_label,
      }
      yield idx, record



class AnnParser():
  """Deeplesion Annotation Parser
  Attributes:
    ann_path: `str`, path of the annotation file
    config: `tfds.core.BuilderConfig`, builder_config
    ann: `dict`, <split>:`pandas.Dataframe`, parsed annotation
  """
  def __init__(self, ann_path):
    """Inits with path of the annotation file
    """
    self.ann_path = ann_path
  
  @utils.memoized_property
  def ann(self):
    _ann = self._ann_parser()
    return _ann

  def _ann_parser(self):
    """Returns annotions of three splits
      cleanup the annotations,
      
      1. filter manufacturer
      2. take a window 
        ```
        if len(win_level) > 1:
            win_level = np.median(win_level) * 0.985
            win_width = np.max(win_width) * 1.55
        else:
            win_level = win_level[0]
            win_width = win_width[0]
        ```
      3. map lesion type
      4. remove bbox associate with calc
      5. split by Split_Num
    """
    pd = tfds.core.lazy_imports.pandas
    with tf.io.gfile.GFile(self.ann_path) as csv_f:
      df = pd.read_csv(csv_f)

    # select columns
    df_t = df[['fake id', 'full path', 'window center',
              'window width', 'rows', 'columns', 'manufacturer',
              'lesion type', 'bounding box', 'Split_Num', "Class", "Result"
              ]]
    def format_win_center(s):
      win_center = [int(a) for a in s.strip('[]').split(', ')]
      if len(win_center) > 1:
          win_center = np.median(win_center) * 0.985
      else:
          win_center = win_center[0]
      return win_center

    def format_win_width(s):
        win_width = [int(a) for a in s.strip('[]').split(', ')]
        if len(win_width) > 1:
            win_width = np.max(win_width) * 1.55
        else:
            win_width = win_width[0]
        return win_width

    def remove_element(r, col_name):
        idx_2rm = r['idx_2rm']
        for i in sorted(idx_2rm, reverse = True):
            try:
                del r[col_name][i]
            except IndexError as e:      
                raise e
        return r[col_name]

    df_t = df_t[df_t['manufacturer'] == 'GE MEDICAL SYSTEMS']
    df_t['full path'] = df_t['full path'].apply(lambda x: '/'.join(x.split('/')[1:]))
    df_t['window center'] = df_t['window center'].apply(format_win_center)
    df_t['window width'] = df_t['window width'].apply(format_win_width)
    df_t = df_t[(df_t['lesion type'] != '[10000]') & (df_t['lesion type'] != '[10000, 10000]') & (df_t['lesion type'] != '[10000, 10000, 10000]')]
    df_t['lesion type'] = df_t['lesion type'].apply(lambda x: [LESION_TYPES[a] for a in x.strip('[]').split(', ')])
    df_t['idx_2rm'] = df_t['lesion type'].apply(lambda x: [i for i, a in enumerate(x) if a=='calc'])
    df_t['lesion type'] = df_t.apply(lambda x: remove_element(x, 'lesion type'), axis=1)
    df_t['bounding box'] = df_t['bounding box'].apply(lambda x: ast.literal_eval(x))
    #df_t = df_t[df_t['idx_2rm'].map(lambda x: len(x) >= 1) ]
    df_t['bounding box'] = df_t.apply(lambda x: remove_element(x, 'bounding box'), axis=1)
    df_t = df_t.drop('idx_2rm', axis=1)
    df_t['Class'] = df_t['Class'].replace('HR benign', 'Benign')
    df_t['Result'] = df_t['Result'].replace('TN', 'FP')
    df_t['Result'] = df_t['Result'].replace('FN', 'TP')
    # split
    return {'train': df_t[df_t['Split_Num'] < 0.7],
            'validation': df_t[df_t['Split_Num'] >=0.7],
            }