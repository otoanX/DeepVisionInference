# ChatGPT
import os
import json
from flask import Flask, render_template, request, jsonify
import base64


app = Flask(__name__)

IMAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def get_latest_image():
    image_files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.endswith(('.jpg', '.png', '.gif'))]
    if image_files:
        latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(IMAGE_DIRECTORY, x)))
        return latest_image
    else:
        return None


def srcBase64(image_path):
    if image_path.endswith(('.jpg', '.png', '.gif')):
        image_ext = os.path.splitext(image_path)[1][1:]
        with open(image_path, 'rb') as f:
            image_base64 = str(base64.b64encode(f.read()), 'utf-8')
        return "data:image/" + image_ext + ";base64," + image_base64
    else:
        return None


@app.route('/')
def display_latest_image():
    latest_image = get_latest_image()
    if latest_image:
        image_path = os.path.join(IMAGE_DIRECTORY, latest_image)
        return render_template('index.html', image_path=srcBase64(image_path))
    else:
        return "ディレクトリ内に画像ファイルが見つかりません。"


@app.route('/latest_image', methods=['GET'])
def get_latest_image_json():
    latest_image = get_latest_image()
    if latest_image:
        image_path = os.path.join(IMAGE_DIRECTORY, latest_image)
        # print(image_path)
        return jsonify({"image_path": srcBase64(image_path)})
    else:
        return jsonify({"error": "ディレクトリ内に画像ファイルが見つかりません。"})


if __name__ == '__main__':
    app.run(debug=True)
