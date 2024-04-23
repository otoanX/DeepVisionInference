import subprocess
import time


# print("学習")
# subprocess.run(["python", "./src/api/mnist_pytorch.py"])

print("サーバー起動")
server = subprocess.Popen(["python", "./src/api/tm_server.py"])
time.sleep(10)

print("サーバーに画像を送信")
subprocess.run(["python", "./src/api/tm_send.py"])

print("サーバー立ち下げ")
server.kill()

print("完了")
