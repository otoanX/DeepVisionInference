import base64
from PIL import Image
import io
import json
import flask
from flask import request

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import MNIST


# ポート番号
TM_PORT_NO = 8085
# HTTPサーバーを起動
app = flask.Flask(__name__)
print(F"http://localhost:{TM_PORT_NO}")


# Base64形式の文字列から画像データを取得する関数
def base64_to_image(base64_string):
    base64_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(base64_data))
    return image


# 画像の前処理
def preprocess_image(base64_image):
    # Base64形式の画像データをPIL Imageに変換
    image = base64_to_image(base64_image)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # グレースケールに変換
        transforms.Resize((28, 28)),  # 28x28にリサイズ
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# /apiへアクセスした場合
@app.route('/api', methods=['GET'])
def api():
    # URLパラメータを取得
    image = request.args.get('image', '')
    if image == '':
        return '{"score": "空です"}'
    print(f"画像を取得したので処理します")
    # ここに判定処理を記述
    outputs = model(preprocess_image(image))
    _, score = torch.max(outputs, 1)
    # return print(f"score: {score.item()}")
    answer = json.dumps({
        # "score": score,
        "item": score.item()
    })
    return answer


if __name__ == "__main__":
    # 学習済みモデルのロード
    model = CNN()
    # 学習済みモデルのパラメータを読み込む
    model.load_state_dict(torch.load('./mnist_cnn_model.pth'))
    model.eval()  # 推論モードに設定
    # サーバーを起動
    app.run(debug=False, port=TM_PORT_NO)
