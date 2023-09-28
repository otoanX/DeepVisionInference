
import asyncio
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
 
import time


class ImageInference(object):
    """
    画像の推論を実施するクラス
    .img(変数): BGR画像を読み込む
    .read_img(imgpath): 画像ファイルを読み込む
    """
    def __init__(self):
        pass

    def __del__(self):
        pass

    def img(self, image):
        self.image = image

    def read_img(self, imgpath):
        self.image = cv2.imread(imgpath)

    def tone_correction(self, alpha, beta):  # 色調補正(コントラスト、明るさ)
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)

    def gray_scale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    # ファイル／フォルダの変化があった時に呼ばれるイベントハンドラ
class EventHandler(FileSystemEventHandler):
    def on_created(self, e):   # 変化があったときにさせたい処理
        # print(f"{e.is_directory} : {e.event_type} : {e.src_path}")
        # is_directory	変化が起きた対象がディレクトリの場合はTrue、ファイルの場合はFalse
        # event_type	変化の内容が文字列として渡される
        # “modified” →変更
        # “created”　→作成
        # “deleted”　→削除
        # “moved”　→移動
        # src_path	変化が起きたフォルダまたはファイルのパス
        if e.is_directory is False:
            return
        if e.event_type != "created":
            return


if __name__ == "__main__":
    # ファイル／フォルダの監視を開始
    observer = Observer()
    WATCH_DIR = "./"
    observer.schedule(EventHandler(), path=WATCH_DIR, recursive=False)
    observer.start()
    
    # 監視のためのループ処理
    try:
        while True:
            time.sleep(1)
    except:
        observer.join()
