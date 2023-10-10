import atexit
import ctypes
import os
import time
import torch
from torchvision import transforms, models
from PIL import Image
# import sqlite3
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import keyboard
from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG


# ロギング設定
formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = StreamHandler()
# file_handler = FileHandler("log_inference.log")
file_handler.setFormatter(formatter)

log_inference = getLogger(__name__)
log_inference.addHandler(file_handler)
log_inference.setLevel(DEBUG)
log_inference.debug("Start inference.py")
def atexitLog():
    log_inference.debug("END inference.py")
atexit.register(atexitLog)

# モデルのロード
# model = torch.load('your_model.pth')  # 自作モデル
# model = models.vgg16(pretrained=True)   # vgg16
weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1
model = models.efficientnet_b1(weights=weights)
model.eval()

# 画像を変換するための前処理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# データベース接続の設定
# connection = sqlite3.connect('your_database.db')


# 新しい画像を処理する関数
def process_image(image_path):
    try:
        # 画像を前処理
        image = Image.open(image_path)
        image = preprocess(image)
        image = image.unsqueeze(0)  # バッチ次元を追加

        # 推論
        with torch.no_grad():
            output = model(image)
        log_inference.debug(
            "image path: " + image_path +
            ", class No: " + str(output.argmax()) +
            ", predict: " + str(output.max()))

        # 推論結果をデータベースに保存
        # cursor = connection.cursor()
        # cursor.execute("INSERT INTO results (image_path, class, confidence) VALUES (?, ?, ?)",
        #                (image_path, output.argmax(), output.max()))
        # connection.commit()

        # 画像を処理済みとして移動または削除する
        # os.remove(image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")


# フォルダの変更を監視するハンドラクラス
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.jpg'):
            file_writed_wait(event.src_path)
            process_image(event.src_path)


def file_writed_wait(path):
    while True:
        try:
            new_path = path + "_"
            os.rename(path, new_path)
            os.rename(new_path, path)
            time.sleep(0.05)
            break
        except OSError:
            time.sleep(0.05)
        return

# フォルダの監視を開始
image_folder = os.path.join(os.path.dirname(__file__), 'image')
event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path=image_folder, recursive=True)
observer.start()


# キーボードイベントを監視してエスケープキーが押されたら終了
class check_exit_key(threading.Thread):
    def __init__(self):
        super().__init__()
        self.signal = threading.Event()
        self.alive = True

    def run(self):
        while self.alive:
            if keyboard.is_pressed('esc'):
                log_inference.debug("Push ESC")
                self.signal.set()
            time.sleep(0.1)

    def kill(self):
        self.alive = False

    def exit_signal(self):
        return self.signal

    def clear_signal(self):
        self.signal.clear()


# エスケープキー監視スレッドを開始
# exit_signal = threading.Event()  # スレッド終了シグナル
# exit_thread = threading.Thread(target=check_exit_key, daemon=True)
# exit_thread.start()
## kill機能追加スレッド
exit_thread = check_exit_key()
exit_thread.start()

try:
    # while not exit_signal.is_set():
    print("Target dir: ", image_folder)
    while True:
    # while observer.is_alive():
        if exit_thread.exit_signal().is_set():
            log_inference.debug("exit_signal is set")
            observer.stop()
            observer.join()
            log_inference.debug("Observer Joined")
            observer = Observer()
            observer.schedule(event_handler, path=image_folder, recursive=True)
            observer.start()
            log_inference.debug("Observer Restarted")
            exit_thread.clear_signal()
        time.sleep(1)
except KeyboardInterrupt:
    log_inference.debug("KeyboardInterrupted")
    time.sleep(1)

    # フォルダ監視を停止
    observer.stop()
    log_inference.debug("Obserber joining...")
    observer.join()
    log_inference.debug("Observer Joined")

    # キーボードイベントのスレッドが終了するまで待つ
    # exit_thread.join()
    exit_thread.kill()
    log_inference.debug("exit_thread Joined")

    # データベース接続をクローズ
    # connection.close()
