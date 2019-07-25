from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.io.gcp.internal.clients import bigquery

import logging
import random
import string
from subprocess import check_output, CalledProcessError, STDOUT

NUM_WORKERS = 20
REQUIREMENTS_FILE = "requirements.txt"
WORKER_SETUP_FILE = "./setup.py"
MACHINE_TYPE = "n1-standard-1"
NUM_CORES = 1
BEAM_RUNNER = "DataflowRunner"
UNIQUE_ID = [ENTER YOUR NAME HERE]


def run_command(shell_command, throw_error=True):
    try:
        logging.info(shell_command)
        return check_output(shell_command, shell=True, stderr=STDOUT)
    except CalledProcessError as e:
        if throw_error:
            raise ValueError(e.output)
        else:
            logging.error("Error executing `{}`:\n{}".format(shell_command, e.output))


def gcs_download_dir(remote_path, local_path='/tmp/'):
    run_command('gsutil -m cp -r ' + remote_path + ' ' + local_path, throw_error=False)


def get_random_string(N):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


def create_table_schema():
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


def create_pipeline_options():
    pipeline_options = PipelineOptions([
        '--project=tm-ml-platform',
        '--runner={}'.format(BEAM_RUNNER),
        '--temp_location=gs://ml-platform-dataflow-staging/temp',
        '--staging_location=gs://ml-platform-dataflow-staging/staging',
        '--job_name=doggos-predict-by-{}-{}'.format(UNIQUE_ID, get_random_string(6)),
        '--num_workers={}'.format(NUM_WORKERS),
        '--region=us-west1',
        '--autoscaling_algorithm=NONE',
        '--worker_machine_type={}'.format(MACHINE_TYPE),
        '--disk_size_gb=50',
        '--setup_file={}'.format(WORKER_SETUP_FILE)
    ])
    pipeline_options.view_as(SetupOptions).save_main_session = True
    return pipeline_options
