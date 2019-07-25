import logging
import time
from beam_functions import util

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from os.path import abspath
import numpy as np


def predict_images(bq_rows, yaml_config):
    pkey = bq_rows[0]
    image_cols = bq_rows[1]

    # Get images
    logging.info("Getting images")
    imgs_dir = "imgs/{}".format(pkey)
    util.run_command("mkdir -p {}".format(imgs_dir), throw_error=True)
    util.gcs_download_dir(yaml_config['gcs_img_path'].format(pkey), imgs_dir)

    # Load model
    start = time.time()
    model = ResNet50(weights='imagenet')
    end = time.time()
    logging.info("Model loading for breed {} took: {:.2f} sec".format(pkey, end - start))

    # Predicting
    logging.info("Predicting on breed: {}".format(pkey))
    start = time.time()
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
    end = time.time()
    logging.info("Prediction took: {:.2f} sec".format(end-start))

    # Clean up images
    util.run_command("rm -rf {}".format(imgs_dir), throw_error=True)
    return img_predictions
