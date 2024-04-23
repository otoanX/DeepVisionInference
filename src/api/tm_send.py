import requests
import base64
import torch
import torchvision
from torchvision import transforms
# from PIL import Image
import io

# MNISTデータセットをダウンロード
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# データローダーから1つのサンプルを取得
sample_image, _ = next(iter(test_loader))

# PIL画像に変換
pil_image = transforms.ToPILImage()(sample_image.squeeze())

# 画像をBase64エンコード
buffer = io.BytesIO()
pil_image.save(buffer, format='PNG')
encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

# エンドポイントURL
endpoint_url = 'http://localhost:8085/api'

# パラメーターとしてBase64エンコードされた画像データを含むGETリクエストを送信
response = requests.get(endpoint_url, params={'image': encoded_image})

# レスポンスの表示
print(f"tm_send.py response: {response}")
print(f"tm_send.py response.json(): {response.json()}")
