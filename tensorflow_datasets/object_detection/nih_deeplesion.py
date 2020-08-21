"""nih_deeplesion dataset."""

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from tensorflow_datasets.core import utils
import os
import math
import numpy as np
import random

_CITATION = """\
@article{DBLP:journals/corr/abs-1710-01766,
  author    = {Ke Yan and
               Xiaosong Wang and
               Le Lu and
               Ronald M. Summers},
  title     = {DeepLesion: Automated Deep Mining, Categorization and Detection
               of Significant Radiology Image Findings using Large-Scale
               Clinical Lesion Annotations},
  journal   = {CoRR},
  volume    = {abs/1710.01766},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.01766},
  archivePrefix = {arXiv},
  eprint    = {1710.01766},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1710-01766},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

"""

_DESCRIPTION = """\
The DeepLesion dataset contains 32,120 axial computed tomography (CT) slices 
from 10,594 CT scans (studies) of 4,427 unique patients. There are 1–3 lesions 
in each image with accompanying bounding  boxes  and  size  measurements,  
adding  up  to  32,735  lesions  altogether.  The  lesion annotations were 
mined from NIH’s picture archiving and communication system (PACS).
"""

_URL = ("https://nihcc.app.box.com/v/DeepLesion")

class DeeplesionConfig(tfds.core.BuilderConfig):
  """BuilderConfig for DeeplesionConfig."""
  def __init__(self,
               name=None,
               thickness=None,
               **kwargs):
    super(DeeplesionConfig,
          self).__init__(name=name,
                         version=tfds.core.Version('1.0.0'),
                         **kwargs)
    self.thickness = thickness

class NihDeeplesion(tfds.core.GeneratorBasedBuilder):
  """DeepLesion dataset builder."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You have to put the data and the csv file in the
  manual_dir.
  """
  
  BUILDER_CONFIGS = [
    DeeplesionConfig(
      name='abnormal',
      description=_DESCRIPTION,
    ),
    DeeplesionConfig(
      name='normal',
      description=_DESCRIPTION,
    ),
    DeeplesionConfig(
      name='volume',
      description=_DESCRIPTION,
      thickness=5
    ),
  ]

  
  def _info(self):
    if self.builder_config.name == 'abnormal':
      features = tfds.features.FeaturesDict({
          "image/name":
          tfds.features.Text(),
          "image":
          tfds.features.Image(shape=(None, None, 1),
                              dtype=tf.uint16,
                              encoding_format='png'),
          "bboxs":
          tfds.features.Sequence(tfds.features.BBoxFeature()),
      })
    elif self.builder_config.name == 'normal':
      features = tfds.features.FeaturesDict({
          "image/name":
          tfds.features.Text(),
          "image":
          tfds.features.Image(shape=(None, None, 1),
                              dtype=tf.uint16,
                              encoding_format='png'),
      })
    elif self.builder_config.name == 'volume':
      features = tfds.features.FeaturesDict({
          "key_image/name":
          tfds.features.Text(),
          "images":
          tfds.features.Sequence(
              tfds.features.Image(shape=(None, None, 1),
                                  dtype=tf.uint16,
                                  encoding_format='png')),
          "bboxs":
          tfds.features.Sequence(tfds.features.BBoxFeature()),
          "key_index":
          tfds.features.Tensor(shape=(), dtype=tf.int32),
          "measurement_coord":
          tfds.features.Sequence(
              tfds.features.Tensor(shape=(8, ), dtype=tf.float32)),
          "lesion_diameters_pixel":
          tfds.features.Sequence(
              tfds.features.Tensor(shape=(2, ), dtype=tf.float32)),
          "normalized_lesion_loc":
          tfds.features.Sequence(
              tfds.features.Tensor(shape=(3, ), dtype=tf.float32)),
          "corse_lesion_type":
          tfds.features.Sequence(
              tfds.features.Tensor(shape=(), dtype=tf.int32)
          ),
          "possibly_noisy":
          tfds.features.Tensor(shape=(), dtype=tf.int32),
          "slice_range":
          tfds.features.Tensor(shape=(2, ), dtype=tf.int32),
          "slice_range_trunc":
          tfds.features.Tensor(shape=(2, ), dtype=tf.int32),
          "spacing_mm_px":
          tfds.features.Tensor(shape=(3, ), dtype=tf.float32),
          "image_size":
          tfds.features.Tensor(shape=(2, ), dtype=tf.int32),
          "dicom_windows":
          tfds.features.Sequence(
              tfds.features.Tensor(shape=(2, ), dtype=tf.int32)),
          "patient_gender":
          tfds.features.Tensor(shape=(), dtype=tf.int32),
          "patient_age":
          tfds.features.Tensor(shape=(), dtype=tf.int32),
      })
    else:
      raise AssertionError('No builder_config found!')

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=features,
        homepage=_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.manual_dir
    image_path = os.path.join(path, 'Images_png')
    ann_path = os.path.join(path, 'DL_info.csv')
    if not tf.io.gfile.exists(image_path) or not tf.io.gfile.exists(ann_path):
      msg = ("You must download the dataset and annotation file from {} manually and place it into {}.".format(_URL, path))
      raise AssertionError(msg)
    

    # create two helper instances:
    # `fileUtils` to read images
    # `annParser` to parse the annotation file `DL_info.csv`
    annParser = AnnParser(ann_path, config=self.builder_config)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_path": image_path,
                "split": annParser.ann['train'],
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "image_path": image_path,
                "split": annParser.ann['validation'],
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "image_path": image_path,
                "split": annParser.ann['test'],
            },
        ),
    ]


  def _generate_examples(self, image_path, split):
    """Returns examples
    Args:
      archive: `ArchiveUtils`, read image(s) from archives using filename(s)
      split: `pandas.DataFrame`, each row contains an annotation of an image
    Yields:
      example key and data
    """
    if self.builder_config.name in ['abnormal', 'normal']:
      for idx, value in enumerate(split.values):
        # collect annotations
        file_name, bboxs, im_size, _, _, _, _, _, _, _, _, _, _, _, _ = value
        fileNameUtil = FileNameUtils(file_name)
        # build example
        record = {
            "image/name": file_name,
            "image": os.path.join(image_path, fileNameUtil.filePath),
        }
        if self.builder_config.name == 'abnormal':
          record.update(bboxs = _format_bboxs(bboxs, im_size))
        yield idx, record

    elif self.builder_config.name == 'volume':
      for idx, value in enumerate(split.values):
        # collect annotations
        file_name, bboxs, size, ks_idx, m_coords, diameters, n_loc, c_lesion_type, p_noisy, slice_range, spacing, dicom_windows, p_gender, p_age, sp = value

        # generate name list and cutoff indices for tructated series
        fileNameUtil = FileNameUtils(file_name)
        startIdx, endIdx = [int(x) for x in slice_range.split(',')]
        fname_list = fileNameUtil.get_fname_list(startIdx, endIdx)
        t_startIdx, t_endIdx = _cal_cutoff_position(
            len(fname_list), self.builder_config.thickness, ks_idx - startIdx)

        # build example
        record = {
            "key_image/name":
            file_name,
            "images":
            [os.path.join(image_path, FileNameUtils(fname).filePath) for fname in fname_list[t_startIdx:t_endIdx + 1]],
            "bboxs":
            _format_bboxs(bboxs, size),
            "key_index":
            ks_idx - startIdx - t_startIdx,
            "measurement_coord":
            _format_values(m_coords, 8, float),
            "lesion_diameters_pixel":
            _format_values(diameters, 2, float),
            "normalized_lesion_loc":
            _format_values(n_loc, 3, float),
            "corse_lesion_type":
            [int(x) for x in c_lesion_type.split(',')],
            "possibly_noisy":
            int(p_noisy),
            "slice_range": [startIdx, endIdx],
            "slice_range_trunc":
            [t_startIdx + startIdx, t_endIdx + startIdx],
            "spacing_mm_px": [float(x) for x in spacing.split(',')],
            "image_size": [int(x) for x in size.split(',')],
            "dicom_windows":
            _format_values(dicom_windows, 2, int),
            "patient_gender":
            int(p_gender),
            "patient_age":
            int(p_age),
        }
        yield idx, record

    else:
      raise AssertionError('No builder_config found!')


class FileNameUtils():
  """Helper class to parse fileName from the annotation file
  Attributes:
    fileName: `str`, "<seriesFolder>_<%03d>.png"
    seriesFolder: `str`, "<seriesFolder>"
    sliceIdx: `int`, d
    sliceFileName: `str`, "<%03d>.png"
    filePath: `str`, "<seriesFolder>/<%03d>.png"
  """
  def __init__(self, fileName):
    """Inits with a fileName"""
    self.fileName = fileName

  @utils.memoized_property
  def seriesFolder(self):
    return '_'.join(self.fileName.split('_')[:-1])

  @utils.memoized_property
  def sliceIdx(self):
    return int(self.fileName.split('_')[-1][:-4])

  @utils.memoized_property
  def sliceFileName(self):
    return '%03d.png' % self.sliceIdx

  @utils.memoized_property
  def filePath(self):
    return os.path.join(self.seriesFolder, self.sliceFileName)


  def get_fname(self, sliceIdx):
    """Returns a fileName in annotated format
    Args:
      sliceIdx: `int`, index of a slice (the index should in valid slice_range)
    Returns:
      a `str` in "<seriesFolder>_<%03d>.png" format
    """
    return '_'.join([self.seriesFolder, '%03d.png' % sliceIdx])

  def get_fname_list(self, start, end):
    """Returns a list of names contructed by continuous indices
    Args:
      start: `int`, start point of continuous indices
      end: `int`, end point of continuous indices (inclusive)
    Returns:
      a list of `str`
    """
    return [self.get_fname(s) for s in range(start, end + 1)]


class AnnParser():
  """Deeplesion Annotation Parser
  Attributes:
    ann_path: `str`, path of the annotation file
    config: `tfds.core.BuilderConfig`, builder_config
    ann: `dict`, <split>:`pandas.Dataframe`, parsed annotation
  """
  def __init__(self, ann_path, config=None):
    """Inits with path of the annotation file
    """
    self.ann_path = ann_path
    self.config = config
  
  @utils.memoized_property
  def ann(self):
    _ann = self._ann_parser()
    if self.config.name == 'normal':
      _ann = {'train': self._create_ann_for_normals(_ann['train']),
              'validation': self._create_ann_for_normals(_ann['validation']),
              'test': self._create_ann_for_normals(_ann['test']),
             }
    return _ann

  def _ann_parser(self):
    """Returns annotions of three splits
      cleanup the annotations,
      group the annotations by File_name,
      split the annotations by Train_Val_Test
    """
    pd = tfds.core.lazy_imports.pandas
    with tf.io.gfile.GFile(self.ann_path) as csv_f:
        # read file
      df = pd.read_csv(csv_f)

      # select columns
      df_t = df[['File_name', 'Bounding_boxes', 'Image_size',
                'Key_slice_index', 'Measurement_coordinates', 'Lesion_diameters_Pixel_',
                'Normalized_lesion_location', 'Coarse_lesion_type', 'Possibly_noisy', 'Slice_range',
                'Spacing_mm_px_', 'DICOM_windows', 'Patient_gender',
                'Patient_age', 'Train_Val_Test']]
      df_t = df_t.copy()

      # clean data
      df_t.fillna(-1, inplace=True)
      df_t.Patient_gender.replace(['M', 'F'], [1, 0], inplace=True)
      df_t.Coarse_lesion_type.replace([1,2,3,4,5,6,7,8,-1], ['1','2','3','4','5','6','7','8','-1'], inplace=True)

      mask = df_t.apply(
          lambda x: x['Image_size'].split(", ")[0] == '512', axis=1)
      df_t = df_t[mask]  # filter out Image_size != 512
      df_t = df_t[df_t.Possibly_noisy ==
                  0]  # filter out possibly noisy image
      df_t = df_t[df_t.Spacing_mm_px_ >
                  "0.6"]  # filter out spacing mm/pixel < 0.6
      space = df_t.Spacing_mm_px_.apply(
          lambda x: float(x.split(", ")[2]))
      df_t['z_space'] = space
      df_t = df_t[(df_t.z_space <= 6) & (df_t.z_space >= 1)]
      df_t.drop(columns=['z_space'], inplace=True)

      # group and aggregate
      def concat(a): return ", ".join(a)  # rules for aggregation
      d = {
          'Bounding_boxes': concat,
          'Image_size': 'first',
          'Key_slice_index': 'first',
          'Measurement_coordinates': concat,
          'Lesion_diameters_Pixel_': concat,
          'Normalized_lesion_location': concat,
          'Coarse_lesion_type': concat,
          'Possibly_noisy': 'first',
          'Slice_range': 'first',
          'Spacing_mm_px_': 'first',
          'DICOM_windows': concat,
          'Patient_gender': 'first',
          'Patient_age': 'first',
          'Train_Val_Test': 'first',
      }
      df_new = df_t.groupby(
          'File_name',
          as_index=False).aggregate(d).reindex(columns=df_t.columns)

      df_new = df_new.drop_duplicates("File_name")

      # split
      return {'train': df_new[df_new['Train_Val_Test'] == 1],
              'validation': df_new[df_new['Train_Val_Test'] == 2],
              'test': df_new[df_new['Train_Val_Test'] == 3]
             }

  def _create_ann_for_normals(self, ann):
    """Returns new created dataframe of normal scans
    To calulate the length of abnormal area in z direction
    known: spacing (mm per pixel) in z direction, and the length
      of long axis of lesion(pixels)
    assumption: assume lesions are enclosed in a ball-shape area,
      and the long axis is the longest axis in all direction.
    1. convert length of long axis from pixels into mm using
      spacing (mm per pixel) in x-y plane
    2. take this length as the abnormal range in z direction, convert
      it into num of slice 
    Args:
      ann: `pandas.Dataframe`, annotations of abnormal scans
    Returns:
      a `pandas.Dataframe`, columns align with ann
    """
    normal_scans_ann = []  # list of dicts to initiate a new dataframe
    for idx, value in enumerate(ann.values):
      # collect info from each row
      file_name, bboxs, size, ks_idx, m_coords, diameters, n_loc, c_lesion_type, p_noisy, slice_range, spacing, dicom_windows, p_gender, p_age, sp = value

      # calculate (approximately) offset of normal area from key slice 
      # spacing in x, y, z direction, (float, float, float)
      spacing_xy_mm_px = float(spacing.split(',')[0])  
      spacing_z_mm_interval = float(spacing.split(',')[2])
      longest_px = max([
          float(x) for x in diameters.split(',')
      ])  # the first one is always the longest, list of (float, float)
      longest_mm = longest_px * spacing_xy_mm_px
      offset_z = math.ceil(longest_mm / spacing_z_mm_interval)

      # randomly pick out a normal scan
      s_range = [int(x) for x in slice_range.split(',')]
      slice_idxs = [
          x for x in range(s_range[0], s_range[1] + 1)
      ]  # collect all valid idxs within the boundaries (includsively)
      key_idx = int(ks_idx)
      valid_idxs_pool = list(
          filter(
              lambda x: True
              if abs(x - key_idx) > offset_z else False, slice_idxs))

      # create a record for the normal scan
      for normal_idx in valid_idxs_pool:
        normal = {
            'File_name':
            FileNameUtils(file_name).get_fname(normal_idx),
            'Bounding_boxes': None,
            'Image_size': size,
            'Key_slice_index': ks_idx,
            'Measurement_coordinates': None,
            'Lesion_diameters_Pixel_': None,
            'Normalized_lesion_location': None,
            "Coarse_lesion_type": None,
            'Possibly_noisy': p_noisy,
            'Slice_range': slice_range,
            'Spacing_mm_px_': spacing,
            'DICOM_windows': dicom_windows,
            'Patient_gender': p_gender,
            'Patient_age': p_age,
            'Train_Val_Test': sp,
        }
        normal_scans_ann.append(normal)

    # create a new Dataframe of normals
    pd = tfds.core.lazy_imports.pandas
    normal_ann = pd.DataFrame(
        normal_scans_ann,
        columns=[
            'File_name', 'Bounding_boxes', 'Image_size', 'Key_slice_index',
            'Measurement_coordinates', 'Lesion_diameters_Pixel_',
            'Normalized_lesion_location', 'Coarse_lesion_type', 'Possibly_noisy', 'Slice_range',
            'Spacing_mm_px_', 'DICOM_windows', 'Patient_gender',
            'Patient_age', 'Train_Val_Test',
        ])
    normal_ann = normal_ann.drop_duplicates("File_name")
    mask = normal_ann["File_name"].isin(ann["File_name"])
    normal_ann = normal_ann[~mask]

    return normal_ann.sample(n=len(ann), random_state=42)


def _cal_cutoff_position(length, thickness, keyIdx):
  """Returns cutoff indices of keyIdx-centred truncated list
  Args:
    length: `int`, length of a list
    thickness: `int`, cutoff thickness - number of slices
    keyIdx: `int`, index of the key slice
  Returns:
    a tuple of two `int`
  """
  left_block = (thickness - 1) // 2
  right_block = thickness - 1 - left_block

  start = max(0, keyIdx - left_block)
  end = min(length - 1, keyIdx + right_block)
  return start, end


def _format_bboxs(bboxs, size):
  """Returns bbox feature
  Args:
    bboxs: `str`, "xmin,ymin,xmax,ymax"
    size: `str`, "height,width", height is assumed to be equal with width.
  Returns:
    tfds.features.BBox
  """
  size = [float(x) for x in size.split(',')]
  if len(size) != 2 or size[0] != size[1]:
    raise AssertionError(
        'height should be equal with width for this dataset')
  coords = np.clip([float(x) for x in bboxs.split(',')], 0.0, size[0])

  cnt = int((len(coords)) / 4)

  bbox_list = []
  for i in range(cnt):
    ymin = coords[i * 4 + 1] / size[1]
    xmin = coords[i * 4] / size[0]
    ymax = coords[i * 4 + 3] / size[1]
    xmax = coords[i * 4 + 2] / size[0]
    bbox_list.append(
        tfds.features.BBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))

  return bbox_list


def _format_values(val, n, e_type):
  """Returns a feature formated value
  Args:
    val: `str`, "num1,num2,num3,...", value to format
    n: `int`, number of elememts within tuples
    e_type: `str`, the type of each elements within tuples
  Returns
    a list of tuples of e_type values
  """
  lst = [e_type(float(x)) for x in val.split(',')]
  return [lst[i:i+n] for i in range(0, len(lst), n)]
