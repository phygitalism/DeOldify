# import the necessary packages
import os
import logging

from flask import Flask
from flask import request
from flask import send_file
import torch

from app_utils import download
from app_utils import generate_random_filename
from app_utils import clean_all
from app_utils import create_directory
from app_utils import get_model_bin
from app_utils import convertToJPG

from deoldify.visualize import *


# Handle switch between GPU and CPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    del os.environ["CUDA_VISIBLE_DEVICES"]


app = Flask(__name__)
app.logger.setLevel(logging.INFO)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# define a predict function as an endpoint
@app.route("/process", methods=["POST"])
def process_image():

    input_path = generate_random_filename(upload_directory,"jpeg")
    output_path = os.path.join(results_img_directory, os.path.basename(input_path))

    try:
        app.logger.info(request.form.getlist("render_factor"))
        if 'file' in request.files:
            file = request.files['file']
            app.logger.info(file)
            if allowed_file(file.filename):
                file.save(input_path)
            try:
                render_factor = request.form.getlist('render_factor')[0]
            except Exception as exc:
                app.log_exception(exc)
                render_factor = 30
            
        else:
            url = request.json["url"]
            download(url, input_path)

            try:
                render_factor = request.json["render_factor"]
            except Exception as exc:
                app.log_exception(exc)
                render_factor = 30

        try:
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20,20),
                render_factor=int(render_factor), display_render_factor=True, compare=False)
        except:
            convertToJPG(input_path)
            image_colorizer.plot_transformed_image(path=input_path, figsize=(20,20),
            render_factor=int(render_factor), display_render_factor=True, compare=False)

        callback = send_file(output_path, mimetype='image/jpeg')
        
        return callback, 200

    except Exception as exc:
        app.log_exception(exc)
        return {'message': 'input error'}, 400

    finally:
        clean_all([
            input_path,
            output_path
            ])

if __name__ == '__main__':
    global upload_directory
    global results_img_directory
    global image_colorizer
    global ALLOWED_EXTENSIONS
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

    upload_directory = '/data/upload/'
    create_directory(upload_directory)

    results_img_directory = '/data/result_images/'
    create_directory(results_img_directory)

    model_directory = '/data/models/'
    create_directory(model_directory)

    artistic_model_url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"

    # only get the model binay if it not present in /data/models
    get_model_bin(
        artistic_model_url, os.path.join(model_directory, "ColorizeArtistic_gen.pth")
    )

    image_colorizer = get_image_colorizer(artistic=True)

    port = 5000
    host = "0.0.0.0"

    app.run(host=host, port=port, threaded=False)
