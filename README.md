# DeepVisionInference
深層学習を用いた画像推論システム。こんな感じにしたい
```
flaskのウェブ上から下記を設定する
①画像の取得方法
　ファイル/フォルダ選択
　フォルダ監視
②画像の前処理(複数選択可能)
　明るさ修正
　2値化
　トリミング
　リサイズ
　正方形化
③学習済みモデルの選択
　kerasモデル
　pviモデル
　その他API
④分類後の処理
　②に戻るかどうか
　PLCへの伝文
　タイムアウト
実行ボタンを押したら実行される
```
Flaskの初期画面は下記を参考にする
> Flaskを使用し、外部ファイルを実行する  
> https://qiita.com/xxPowderxx/items/6740562e4be87af40e33

## 実行の方法
次の３つのコマンドでテストプログラムが実行できます。

```cmd:コマンドプロンプト
$ docker-compose build
$ docker-compose up -d
$ docker exec -it python3 python test.py
```