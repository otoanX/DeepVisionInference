import tkinter as tk
import os
import base64
import requests
from PIL import Image, ImageTk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from pymodbus.client.sync import ModbusTcpClient
import argparse
import time


def parse_args():
    """
    コマンドライン引数をパースする関数。

    Returns:
        argparse.Namespace: パースされた引数を格納したNamespaceオブジェクト。
    """
    parser = argparse.ArgumentParser(description="Image Viewer and PLC Monitor")
    parser.add_argument("--directory", type=str, default="/path/to/your/directory",
                        help="Path to the directory to monitor for new images")
    parser.add_argument("--plc_ip", type=str, default="localhost", help="IP address of the PLC to monitor")
    parser.add_argument("--endpoint", type=str, default="http://localhost:1000", help="Endpoint to send images")
    parser.add_argument("--response", type=str, default="指定のレスポンス", help="Expected response from the server")
    return parser.parse_args()


# グローバル変数として最新の画像パスを保持するための共有メモリ
latest_image_path = None


class ImageHandler(FileSystemEventHandler):
    def __init__(self, window, canvas):
        super().__init__()
        self.window = window
        self.canvas = canvas

    def on_created(self, event):
        """
        ファイルが作成されたときに呼び出されるコールバック関数。

        Args:
            event (FileSystemEvent): ファイルシステムイベントオブジェクト。
        """
        global latest_image_path
        if event.is_directory or not event.src_path.endswith(('.jpg', '.png', '.jpeg')):
            return
        latest_image_path = event.src_path  # 最新の画像パスを更新
        self.show_image(latest_image_path)

    def show_image(self, path):
        """
        画像をキャンバス上に表示する関数。

        Args:
            path (str): 表示する画像ファイルのパス。
        """
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # 画像が消えないように参照を保持


def send_image_to_endpoint(endpoint):
    """
    エンドポイントに画像を送信する関数。

    Args:
        endpoint (str): 画像を送信するエンドポイントのURL。

    Returns:
        str: サーバーからのレスポンス。
    """
    global latest_image_path
    if latest_image_path:
        # 最新の画像が存在する場合は、その画像をbase64形式にエンコードしてエンドポイントに送信
        with open(latest_image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data)
            response = requests.get(endpoint, params={"image": image_base64})
            print("Response from server:", response.text)
            return response.text


def handle_matching_response(plc_client):
    """
    指定のレスポンスを処理する関数。コイル1と2を1秒間ONにします。

    Args:
        plc_client: Modbus TCPクライアント。
    """
    plc_client.write_coils(1, [True])  # コイル1をONにする
    plc_client.write_coils(2, [True])  # コイル2をONにする
    time.sleep(1)
    plc_client.write_coils(1, [False])  # コイル1をOFFにする
    plc_client.write_coils(2, [False])  # コイル2をOFFにする


def handle_nonmatching_response(plc_client):
    """
    指定のレスポンス以外を処理する関数。コイル1と3を1秒間ONにします。

    Args:
        plc_client: Modbus TCPクライアント。
    """
    plc_client.write_coils(1, [True])  # コイル1をONにする
    plc_client.write_coils(3, [True])  # コイル3をONにする
    time.sleep(1)
    plc_client.write_coils(1, [False])  # コイル1をOFFにする
    plc_client.write_coils(3, [False])  # コイル3をOFFにする


def plc_monitor(plc_client, endpoint):
    """
    PLCを監視し、特定のビットがONになった場合に画像を送信し、レスポンスに応じてコイルの状態を制御する関数。

    Args:
        plc_client: Modbus TCPクライアント。
        endpoint (str): 画像を送信するエンドポイントのURL。
    """
    # 画像が送信されたかどうかを追跡するフラグ
    image_sent = False

    # PLCの特定のビットを監視するループ
    while True:
        # ビットの状態を取得する
        result = plc_client.read_coils(0, 1)  # ここではアドレス0のビットを読み取ります
        if result.bits[0] and not image_sent:  # ビットがONになり、かつ画像がまだ送信されていない場合
            print("特定のビットがONになりました")
            response = send_image_to_endpoint(endpoint)
            if response == args.response:
                handle_matching_response(plc_client)
            else:
                handle_nonmatching_response(plc_client)
            image_sent = True  # 画像が送信されたことをフラグに記録する
        elif not result.bits[0]:  # ビットがOFFになった場合
            image_sent = False  # フラグをリセットする


def main(args):
    """
    メイン関数。Tkinter GUIを初期化し、ファイル監視とPLC監視のスレッドを開始します。

    Args:
        args: コマンドライン引数のNamespaceオブジェクト。
    """
    root = tk.Tk()
    root.title("Image Viewer")

    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()

    event_handler = ImageHandler(root, canvas)
    observer = Observer()
    observer.schedule(event_handler, args.directory, recursive=False)
    observer.start()

    # Modbus TCPクライアントの作成
    plc_client = ModbusTcpClient(args.plc_ip)

    # PLC監視のためのスレッドを開始
    plc_thread = threading.Thread(target=plc_monitor, args=(plc_client, args.endpoint))
    plc_thread.daemon = True
    plc_thread.start()

    root.mainloop()


if __name__ == "__main__":
    args = parse_args()
    main(args)
