"""Classify breed from dog images

Usage:
  predict.py <yaml_config>

Uses pretrained Keras model (ResNet50) to classify dog breed from image files in GCS.

Arguments:
  yaml_config           YAML config file for the job
"""

# Import libraries
from beam_functions import util, predict
import apache_beam as beam
import yaml
from docopt import docopt
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def run_pipeline(yaml_config):
    pipeline_options = util.create_pipeline_options()

    with beam.Pipeline(options=pipeline_options) as p:
        result = (p
                  | 'Read BQ Table' >> beam.io.Read(
                    beam.io.BigQuerySource(query=yaml_config['query'], use_standard_sql=True))
                  | 'Add Image Key' >> beam.Map(lambda row: (row['breed'], row))
                  | 'Group By Image' >> beam.GroupByKey()
                  | 'Run Predictions' >> beam.FlatMap(predict.predict_images, yaml_config=yaml_config)
                  | 'Write BQ Table' >> beam.io.Write(
                    beam.io.BigQuerySink(
                        "{project}:{dataset}.{table}".format(**yaml_config['output']),
                        schema=util.create_table_schema(),
                        write_disposition=beam.io.BigQueryDisposition.WRITE_EMPTY,
                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)))


if __name__ == '__main__':
    assert sys.version_info[0] == 2, "Python 2 needed"
    yaml_path = docopt(__doc__)['<yaml_config>']
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    run_pipeline(yaml_config)
