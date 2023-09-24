from flask import Flask, render_template, request, Response
import threading
import webbrowser
# ↓　Flaskを通し実行したいファイルをインポート
# import inference
from imagepredictlib import Camera, Cameras


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('layout.html', title='Deep Vision Inference')


# ↓ /inferenceをGETメソッドで受け取った時の処理
@app.route('/inference')
def get():
    # チェックボックスの値を得る
    getimglist = request.args.getlist("getimg")
    print("getimglist: ", getimglist)
    return getimglist
    # ↓　実行したいファイルの関数
    # return inference.main()


# @app.route("/stream")
# def stream():
#     return render_template("stream.html")


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + frame.tobytes()
                   + b"\r\n")
        else:
            print("frame is none")


@app.route("/video_feed")
def video_feed():
    return Response(gen(Cameras()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # コードを変更するたびにmainが再実行されるため、その度にブラウザーが追加で開いてしまう
    threading.Timer(1.0, lambda: webbrowser.open('http://localhost:5000') ).start()
    # app.run(debug=True)
    app.run(debug=False)    # コード変更ごとに追加ブラウザが開くのを防ぐため
