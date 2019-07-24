"""Classify breed from dog images

Usage:
  predict.py <yaml_config>

Uses pretrained Keras model (ResNet50) to classify dog breed from image files in GCS.

Arguments:
  yaml_config           YAML config file for the job
"""

# Import libraries
import apache_beam as beam
import yaml

import logging
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.gcp.internal.clients import bigquery
from docopt import docopt

import random
import string
import sys
from subprocess import check_output, CalledProcessError, STDOUT

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from os.path import abspath

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

NUM_WORKERS = 20
REQUIREMENTS_FILE = "requirements.txt"
WORKER_SETUP_FILE = "./setup.py"
MACHINE_TYPE = "n1-standard-1"
NUM_CORES = 1


def _run_command(shell_command, throw_error=True):
    try:
    	logging.info(shell_command)
    	return check_output(shell_command, shell=True, stderr=STDOUT)
    except CalledProcessError as e:
        if throw_error:
            raise ValueError(e.output)
        else:
            logging.error("Error executing `{}`:\n{}".format(shell_command, e.output))


def _gcs_download_dir(remote_path, local_path='/tmp/'):
	_run_command('gsutil -m cp -r ' + remote_path + ' ' + local_path, throw_error=False)


def predict_images(bq_rows, yaml_config):
	pkey = bq_rows[0]
	image_cols = bq_rows[1]

	# Get images
	logging.info("Getting images")
	imgs_dir = "imgs/{}".format(pkey)
	_run_command("mkdir -p {}".format(imgs_dir), throw_error=True)
	_gcs_download_dir(yaml_config['gcs_img_path'].format(pkey), imgs_dir)

	# Load model
	logging.info("Loading model")
	model = ResNet50(weights='imagenet')

	# Predicting
	logging.info("Predicting on images")
	img_predictions = []
	for image_col in image_cols:
		image_key = image_col['url'].split('/')[-1]
		img_path = abspath(imgs_dir + "/" + image_key)

		loaded_img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(loaded_img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		preds = model.predict(x)

		pred_breed = decode_predictions(preds, top=1)[0][0][1]
		score = decode_predictions(preds, top=1)[0][0][2]
		logging.info('For image: {}, predicted: {} with score: {}'.format(image_col['url'], pred_breed, score))
		img_predictions.append({
			'url': image_col['url'],
			'breed': image_col['breed'],
			'prediction': pred_breed,
			'score': str(score)
		})

	# Clean up images
	_run_command("rm -rf {}".format(imgs_dir), throw_error=True)
	return img_predictions


def get_random_string(N):
	return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


def _create_pipeline_options():
    pipeline_options = PipelineOptions([
    	'--project=tm-ml-platform',
    	'--runner=DataflowRunner',
        '--temp_location=gs://ml-platform-dataflow-staging/temp',
        '--staging_location=gs://ml-platform-dataflow-staging/staging',
        '--job_name=doggos-predict-{}'.format(get_random_string(6)),
        '--num_workers={}'.format(NUM_WORKERS),
        '--region=us-west1',
        '--autoscaling_algorithm=NONE',
        '--worker_machine_type={}'.format(MACHINE_TYPE),
        # '--requirements_file={}'.format(REQUIREMENTS_FILE),
        '--disk_size_gb=50',
        '--setup_file={}'.format(WORKER_SETUP_FILE)
    ])
    pipeline_options.view_as(SetupOptions).save_main_session = True
    return pipeline_options


def _create_table_schema():
	table_schema = bigquery.TableSchema()

	url_schema = bigquery.TableFieldSchema()
	url_schema.name = 'url'
	url_schema.type = 'STRING'
	url_schema.mode = 'NULLABLE'
	table_schema.fields.append(url_schema)

	breed_schema = bigquery.TableFieldSchema()
	breed_schema.name = 'breed'
	breed_schema.type = 'STRING'
	breed_schema.mode = 'NULLABLE'
	table_schema.fields.append(breed_schema)

	prediction_schema = bigquery.TableFieldSchema()
	prediction_schema.name = 'prediction'
	prediction_schema.type = 'STRING'
	prediction_schema.mode = 'NULLABLE'
	table_schema.fields.append(prediction_schema)

	score_schema = bigquery.TableFieldSchema()
	score_schema.name = 'score'
	score_schema.type = 'FLOAT64'
	score_schema.mode = 'NULLABLE'
	table_schema.fields.append(score_schema)

	return table_schema


def run_pipeline(yaml_config):
	pipeline_options = _create_pipeline_options()

	with beam.Pipeline(options=pipeline_options) as p:
		result = (p
				  | 'Read BQ Table' >> beam.io.Read(beam.io.BigQuerySource(query=yaml_config['query'], use_standard_sql=True))
				  | 'Add Image Key' >> beam.Map(lambda row: (row['breed'], row))
				  | 'Group By Image' >> beam.GroupByKey()
				  | 'Run Predictions' >> beam.FlatMap(predict_images, yaml_config=yaml_config)
				  | 'Write BQ Table' >> beam.io.Write(
			 		beam.io.BigQuerySink(
			 			"{project}:{dataset}.{table}".format(**yaml_config['output']),
			 			schema=_create_table_schema(),
			 			write_disposition=beam.io.BigQueryDisposition.WRITE_EMPTY,
			 			create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED)))


if __name__ == '__main__':
	yaml_path = docopt(__doc__)['<yaml_config>']
	with open(yaml_path) as f:
		yaml_config = yaml.safe_load(f)

	run_pipeline(yaml_config)
