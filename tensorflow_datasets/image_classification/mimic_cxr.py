"""mimic_cxr dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import numpy as np
import os
import tensorflow_datasets.public_api as tfds
import tensorflow as tf

_CITATION = """
@article{Johnson2019,
abstract = {Chest radiography is an extremely powerful imaging modality, allowing for a detailed inspection of a patient's chest, but requires specialized training for proper interpretation. With the advent of high performance general purpose computer vision algorithms, the accurate automated analysis of chest radiographs is becoming increasingly of interest to researchers. Here we describe MIMIC-CXR, a large dataset of 227,835 imaging studies for 65,379 patients presenting to the Beth Israel Deaconess Medical Center Emergency Department between 2011-2016. Each imaging study can contain one or more images, usually a frontal view and a lateral view. A total of 377,110 images are available in the dataset. Studies are made available with a semi-structured free-text radiology report that describes the radiological findings of the images, written by a practicing radiologist contemporaneously during routine clinical care. All images and reports have been de-identified to protect patient privacy. The dataset is made freely available to facilitate and encourage a wide range of research in computer vision, natural language processing, and clinical data mining.},
author = {Johnson, Alistair E.W. and Pollard, Tom J. and Berkowitz, Seth J. and Greenbaum, Nathaniel R. and Lungren, Matthew P. and Deng, Chih Ying and Mark, Roger G. and Horng, Steven},
doi = {10.1038/s41597-019-0322-0},
file = {:Users/zl190/Downloads/s41597-019-0322-0.pdf:pdf},
issn = {20524463},
journal = {Scientific data},
number = {1},
pages = {317},
pmid = {31831740},
publisher = {Springer US},
title = {{MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports}},
url = {http://dx.doi.org/10.1038/s41597-019-0322-0},
volume = {6},
year = {2019}
}
"""

_DESCRIPTION = """
The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support.

The MIMIC-CXR dataset must be downloaded separately after reading and agreeing 
to a Research Use Agreement. To do so, please follow the instructions on the 
website, https://physionet.org/content/mimic-cxr/2.0.0/
"""


_LABELS = collections.OrderedDict({
  -1.0: "uncertain",
  1.0: "positive",
  0.0: "negative",
  "": "unmentioned",
})


class MimicCxr(tfds.core.BeamBasedBuilder):
  """mimic_cxr dataset."""

  VERSION = tfds.core.Version('0.1.0')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You must register and agree to user agreement on the dataset page:
  https://physionet.org/content/mimic-cxr/2.0.0/
  Afterwards, you have to download and put the mimic-cxr-2.0.0 directory in the
  manual_dir. It should contain subdirectories: files/ with images
  and also mimic-cxr-2.0.0-split.csv, mimic-cxr-2.0.0-negbio.csv,
  mimic-cxr-2.0.0-chexpert.csv, and mimic-cxr-2.0.0-metadata.csv files.
  These four files can be downloaded from https://physionet.org/content/mimic-cxr-jpg/
  """

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          "subject_id": tfds.features.Text(),
          # study related
          "study_id": tfds.features.Text(),
          "studyDate": tfds.features.Text(),
          "studyTime": tfds.features.Text(),
          "report": tfds.features.Text(),
          "performedProcedureStepDescription": tfds.features.Text(),
          "procedureCodeSequence_CodeMeaning": tfds.features.Text(),
          "label_chexpert": tfds.features.Sequence(
                tfds.features.ClassLabel(names=_LABELS.values())),
          "label_negbio": tfds.features.Sequence(
                tfds.features.ClassLabel(names=_LABELS.values())),
          # image related
          "dicom_id": tfds.features.Sequence(tfds.features.Text()),
          "image": tfds.features.Sequence(
              tfds.features.Image(shape=(None, None, 1),
                                  dtype=tf.uint16,
                                  encoding_format='png')),
          "viewPosition": tfds.features.Sequence(tfds.features.Text()),
          "viewCodeSequence_CodeMeaning":tfds.features.Sequence(tfds.features.Text()),
          "patientOrientationCodeSequence_CodeMeaning": tfds.features.Sequence(tfds.features.Text()),
          "rows": tfds.features.Sequence(tfds.features.Tensor(shape=(), dtype=tf.int32)),
          "columns": tfds.features.Sequence(tfds.features.Tensor(shape=(), dtype=tf.int32)),
        }),
        supervised_keys=("image", "label_chexpert"),
        homepage='https://physionet.org/content/mimic-cxr/2.0.0/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.manual_dir
    filesPath = os.path.join(path, 'files')

    if not tf.io.gfile.exists(filesPath):
      msg = ("You must download the dataset folder from MIMIC-CXR"
             "website manually and place it into %s." % path)
      raise AssertionError(msg)

    split_dict = _split_csv_reader(os.path.join(path, 'mimic-cxr-2.0.0-split.csv'))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
              'path': os.path.join(path),
              'split': split_dict['train']
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
              'path': os.path.join(path),
              'split': split_dict['validation']
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
              'path': os.path.join(path),
              'split': split_dict['test']
            },
        ),
    ]

  def _build_pcollection(self, pipeline, path, split):
    """Yields examples."""
    beam = tfds.core.lazy_imports.apache_beam
    pd = tfds.core.lazy_imports.pandas

    # read three csv files
    with tf.io.gfile.GFile(os.path.join(path, 'mimic-cxr-2.0.0-metadata.csv')) as csv_f:
      meta_df = pd.read_csv(csv_f)
      meta_df = meta_df.fillna("")
    with tf.io.gfile.GFile(os.path.join(path, 'mimic-cxr-2.0.0-chexpert.csv')) as csv_f:
      label_chexpert_df = pd.read_csv(csv_f)
      label_chexpert_df = label_chexpert_df.fillna("")
    with tf.io.gfile.GFile(os.path.join(path, 'mimic-cxr-2.0.0-negbio.csv')) as csv_f:
      label_negbio_df = pd.read_csv(csv_f)
      label_negbio_df = label_negbio_df.fillna("")

    chexpert_label_keys = list(label_chexpert_df.columns)[2:]
    negbio_label_keys = list(label_negbio_df.columns)[2:]


    def _extract_content(split, meta_df, label_chexpert_df, label_negbio_df):
      keys = []
      subject_idices = []
      study_idices = []
      dcm_idices = []
      meta_rows = []
      chexpert_rows = []
      negbio_rows = []

      # loop through each unique /p<subject-id>/s<study-id>
      for k, v in split.items():
        # k: files/p<subject-id>[0:3]/p<subject-id>/s<study-id>, v: "dicom-id, ..."
        if k.split(os.path.sep)[1] == 'p10': # test on a small portion of data
          keys.append(k)
          subject_id = k.split(os.path.sep)[2][1:]
          study_id = k.split(os.path.sep)[3][1:]
          dcm_id = [idx.strip() for idx in v[0].split(',')]
          subject_idices.append(subject_id)
          study_idices.append(study_id)
          dcm_idices.append(dcm_id)

          meta_rows.append([meta_df[meta_df.dicom_id == idx] for idx in dcm_id])
          chexpert_rows.append(label_chexpert_df[(label_chexpert_df.subject_id == np.int64(subject_id)) & (label_chexpert_df.study_id == np.int64(study_id))])
          negbio_rows.append(label_negbio_df[(label_negbio_df.subject_id == np.int64(subject_id)) & (label_negbio_df.study_id == np.int64(study_id))])

      return list(zip(keys, subject_idices, study_idices, dcm_idices, meta_rows, chexpert_rows, negbio_rows))


    def _process_example(content):
      Image = tfds.core.lazy_imports.PIL_Image
      pydicom = tfds.core.lazy_imports.pydicom

      k, subject_id, study_id, dcm_id, \
      meta_row, chexpert_row, negbio_row, = content

      with tf.io.gfile.GFile(os.path.join(path, k +'.txt'), 'r') as report:
        report_text = report.read()

      pixelData_list = []
      for idx in dcm_id:
        with tf.io.gfile.GFile(os.path.join(path, k, idx +'.dcm'), 'rb') as dcm:
          image_bytes = dcm.read()
          tmpFile = io.BytesIO(image_bytes)
          image_array = pydicom.dcmread(tmpFile).pixel_array
          image = Image.fromarray(image_array, 'I;16')
          image = image.resize((2544, 3056), resample=Image.NEAREST)
          tmpFile = io.BytesIO()
          image.save(tmpFile, format='PNG')
          tmpFile.seek(0)
        pixelData_list.append(tmpFile)

      record = {
        "subject_id": subject_id,
        # study related
        "study_id": study_id,
        "studyDate": str(meta_row[0].StudyDate.values[0]),
        "studyTime": str(meta_row[0].StudyTime.values[0]),
        "report": report_text,
        "performedProcedureStepDescription": meta_df[meta_df.dicom_id == dcm_id[0]].PerformedProcedureStepDescription.values[0],
        "procedureCodeSequence_CodeMeaning": meta_df[meta_df.dicom_id == dcm_id[0]].ProcedureCodeSequence_CodeMeaning.values[0],
        "label_chexpert": [_LABELS[chexpert_row[key].values[0]] for key in chexpert_label_keys],
        "label_negbio": [_LABELS[negbio_row[key].values[0]] for key in negbio_label_keys],
        # image related
        "dicom_id": dcm_id,
        "image": pixelData_list,
        "viewPosition": [m.ViewPosition.values[0] for m in meta_row],
        "viewCodeSequence_CodeMeaning": [m.ViewCodeSequence_CodeMeaning.values[0] for m in meta_row],
        "patientOrientationCodeSequence_CodeMeaning": [m.PatientOrientationCodeSequence_CodeMeaning.values[0] for m in meta_row],
        "rows": [m.Rows.values[0] for m in meta_row],
        "columns": [m.Columns.values[0] for m in meta_row],
      }
      yield k, record


    return (
        pipeline
        | beam.Create(_extract_content(split, meta_df, label_chexpert_df, label_negbio_df))
        | beam.FlatMap(_process_example)
    )


def _split_csv_reader(split_csv_path):
  """Returns a dictionary {'train/validation/test':{<path>:<'dicom_id, ...'>}}
  """
  pd = tfds.core.lazy_imports.pandas
  with tf.io.gfile.GFile(split_csv_path) as csv_f:
    df = pd.read_csv(csv_f)

    def concat(a): return ", ".join(a)  # rules for aggregation
    df = df.groupby('study_id', as_index=False).aggregate({'subject_id': 'first', 'split': 'first', 'dicom_id': concat})
    df['path'] = df.apply(lambda x: os.path.join("files", 
                            "p" + str(x.subject_id)[0:2], 
                            "p" + str(x.subject_id), 
                            "s" + str(x.study_id)), axis=1)

    return {
      'train': df[df.split=='train'][['path', 'dicom_id']].set_index('path').T.to_dict('list'),
      'validation': df[df.split=='validate'][['path', 'dicom_id']].set_index('path').T.to_dict('list'),
      'test': df[df.split=='test'][['path', 'dicom_id']].set_index('path').T.to_dict('list'),
    }
