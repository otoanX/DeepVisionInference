from flask import Flask, render_template
# ↓　Flaskを通し実行したいファイルをインポート
import inference

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('layout.html', title='Deep Vision Inference')


# ↓ /inferenceをGETメソッドで受け取った時の処理
@app.route('/inference')
def get():
    # ↓　実行したいファイルの関数
    return inference.main()


if __name__ == "__main__":
    app.run(debug=True)
