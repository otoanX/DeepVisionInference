# # FlaskとJavaScriptを使った動的画像表示アプリケーション
# # https://www.s-toki.net/it/post-1133/

# from flask import Flask, send_file, render_template_string
# import os

# app = Flask(__name__)

# # 監視するディレクトリを指定します
# img_directory =</code> "/Users/result" @app.route('/') def index(): return render_template_string(""" &lt;img id="myImage" src="{{ url_for(" /&gt; &lt;script&gt; setInterval(() =&gt; { document.getElementById('myImage').src = '{{ url_for('display_image') }}' + '?' + new Date().getTime(); }, 1000); &lt;/script&gt; """) @app.route('/image') def<code> display_image():
#     # ディレクトリ内の全ファイルをリスト化します
#     files = [os.path.join(img_directory, f) for f in os.listdir(img_directory) if os.path.isfile(os.path.join(img_directory, f))]

#     # ファイルを最終変更時間でソートします
#     files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

#     # ディレクトリ内に画像がある場合、最新の画像を返します
#     if files:
#         return send_file(files[0], mimetype='image/*')
#     else:
#         return "No image found."

# if __name__ == '__main__':
#     # Flaskアプリケーションを起動します
#     app.run(port=8080)


# ChatGPT
import os
import json
from flask import Flask, render_template, request, jsonify
import base64


app = Flask(__name__)

image_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def get_latest_image():
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.gif'))]
    if image_files:
        latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(image_directory, x)))
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
        image_path = os.path.join(image_directory, latest_image)
        return render_template('index.html', image_path=srcBase64(image_path))
    else:
        return "ディレクトリ内に画像ファイルが見つかりません。"


@app.route('/latest_image', methods=['GET'])
def get_latest_image_json():
    latest_image = get_latest_image()
    if latest_image:
        image_path = os.path.join(image_directory, latest_image)
        # print(image_path)
        return jsonify({"image_path": srcBase64(image_path)})
    else:
        return jsonify({"error": "ディレクトリ内に画像ファイルが見つかりません。"})


if __name__ == '__main__':
    app.run(debug=True)
