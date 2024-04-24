import numpy as np
import os
import sys
import time

from PIL import Image
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

sys.path.append(os.path.dirname(__file__))
from plc import udp


class FileChangeHandler(PatternMatchingEventHandler):
    """
    Event handlers for watchdog.
    """
    def __init__(self, owner):
        super().__init__()  # 必須
        self.owner = owner

    def on_created(self, event):
        start_time = time.perf_counter()

        file_path = event.src_path
        if os.path.isdir(file_path):
            return

        # 画像読み込み
        self.img = writed_image_open(file_path)

        # 異品種コーン判定

        # for i, word in enumerate(params.cam):
        #     if word in os.path.basename(file_path):
        #         gui_item[i] = item_check(file_path, self.img)
        #         if item_check(file_path, self.img) != params.cam_items[i]:
        #             print("異品種発生：" + file_path)
        #             print("異品種発生：" + file_path, file=log_file)
        #             gui_data[i] = "-"
        #             if os.path.isdir(ng_move_dir):
        #                 start_date = datetime.datetime.today().strftime("%Y%m%d%H%M%S")[0:8]
        #                 # os.makedirs(ng_move_dir + start_date, exist_ok=True)  # フォルダがなければ作る
        #                 os.makedirs(os.path.join(ng_move_dir, start_date, "item_error"), exist_ok=True)  # フォルダがなければ作る
        #                 shutil.move(file_path, os.path.join(ng_move_dir, start_date, "item_error", file_nm))
        #             return
        #         else:
        #             print("品種：" + item_check(file_path, self.img))

        # AIコーン割れ判定
        with self.session.as_default():
            with self.graph.as_default():
                result = image_predict.predict(self.cnn_model, file_path, self.img)     #AI推論
                ch = result.argmax()
                for i, word in enumerate(params.cam):
                    if word in file_nm:
                        #print("file_nm:" + str(file_nm))
                        #print("i:" + str(i), end="")
                        if ch == 0: # 判定がOKなら
                            datas[i] = "1"  # 正常品でON、排除品でOFF信号(充填機PLC側の仕様)
                            #print(classes[ch] +"の確率：" + result[ch])
                        #print("フラグ：",end="")
                        print(datas)
                        gui_data[i] = classes[ch]
                if ch == 1: # 判定がNGなら
                    if os.path.isdir(ng_move_dir):
                        start_date = datetime.datetime.today().strftime("%Y%m%d%H%M%S")[0:8]
                        os.makedirs(ng_move_dir + start_date, exist_ok=True)  # フォルダがなければ作る
                        shutil.move(file_path, os.path.join(ng_move_dir, start_date, file_nm))
        #計測用 -End
        end_time = time.perf_counter()
        print(f"実行時間:{end_time - start_time}")


def writed_image_open(file_path, timeout=5):
    # HiTriggerによる画像ファイル書き込み完了前にcv読み込みするのを防ぐ。
    # リネームできればファイルアクセス権があることを利用する。
    # 正常にファイル書き込みが完了すればTrueを返す
    # TimeOutで設定した時間以上に書き込みに時間がかかればFalseを返す。
    writestart_time = time.time()
    while True:
        try:
            elapsed_time = time.time()-writestart_time
            new_path = str(file_path)+"_"
            os.rename(file_path, new_path)
            os.rename(new_path, file_path)
            time.sleep(0.01)
            break
        except OSError:
            time.sleep(0.01)
            if elapsed_time >= timeout:
                print("Error:TimeOut")
                return False
    # 画像読み込み
    img = Image.open(file_path)
    if img is None:
        print(f"エラー:Image.openができませんでした。file_path: {file_path}")
        return False
    return img


def item_check(imgpath, img):
    img_mask_average = np.average(img_mask(imgpath, img)[img_mask(imgpath, img)>0])
    for item in params.items_master.keys():
        if params.items_master[item][0] <= img_mask_average < params.items_master[item][1]:
            return item
    return False


def img_mask(imgpath, img):
    # img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE ) # グレースケール
    h, w = img.shape[:2]

    # マスク作成 (黒く塗りつぶす画素の値は0)
    mask = np.zeros((h, w), dtype=np.uint8)
    # 円を描画する関数 circle() を利用してマスクの残したい部分を 255 にしている。
    cv2.circle(mask, center=(w // 2, h // 2), radius=150, color=255, thickness=80)

    # マスクをかける
    img[mask==0] = 0  # mask の値が 0 の画素は黒で塗りつぶす。
    return img

# コマンド実行の確認
if __name__ == "__main__":
    #初期設定
    datas = datas_origin

    # 終了時処理
    atexit.register(pythonexit)

    # ファイル監視の開始
    event_handler = FileChangeHandler([target_file])
    observer = Observer()
    observer.schedule(event_handler, target_dir, recursive=True)
    observer.start()
    print("監視フォルダ：" + target_dir)
    print("監視開始")
    print("モデル名："+model_path)

    # PLC接続
    try:
        print("接続先PLCIPアドレス：" + plc_ip)
        print("CPU型番読み出し...")
        gcplc = udp(plc_ip, port)
    except OSError as error_mes:
        print("PLCに接続できませんでした。通信無しで実行します。")
        print(error_mes)
        plc_disconnect = True

    # 充填機シフトメインループ
    try:
        while True:
            if plc_disconnect is False:
                shift_PLS = gcplc.read("y",20,8)
                if int(shift_PLS) > 0:
                    print("充填機シフトパルス検知")
                    time.sleep(1.0 - pls_deray_time)    #各列画像検査待ち時間
                    predict_answer = ""
                    for i in datas:
                        predict_answer = predict_answer + str(i)
                    gcplc.write("y",1820,str(predict_answer)+"11111111") #検査結果を送信
                    time.sleep(0.1)
                    gcplc.write("y",1820,"0000000000000000") #検査結果を送信
                    print("検査結果送信完了" + str(datas) + str(datetime.datetime.now()))
                    datas = datas_origin  # 初期化
                    print(datas)
                    print(datas_origin)
            time.sleep(0.01)
    except OSError as error_mes:
        print("エラー発生。プログラムを終了します・・・")
        print(error_mes)
    except KeyboardInterrupt:
        print("キー入力がされました。プログラムを終了します・・・")
