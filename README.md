# YOLO26 Small Object Lab

YOLO26がどれくらい小さい物体まで検出できるかを、Androidスマホのカメラで試す環境です。

入力には **android-camera-streamerだけ**を使います。IP Webcam、スマホブラウザ送信、Tailscale Serve、手入力URLには対応しません。

## 起動手順

毎回必要なのは次の4手順だけです。

1. Androidで **android-camera-streamer** を起動する。
2. android-camera-streamerの待受IPに、Tailscaleの `100.x.x.x` を選び、ポート `8080` でサーバーを開始する。
3. PCでTailscaleを起動する。
4. PCで [yolo26_small_object_lab.py](./yolo26_small_object_lab.py) を起動する。

```powershell
python .\yolo26_small_object_lab.py
```

`.py`をダブルクリックしても起動できます。PC側は次を自動実行します。

- Tailscaleの接続状態を確認
- オンラインのAndroid端末を検索
- AndroidのTailscale IPv4アドレスを取得
- `http://<AndroidのTailscale IP>:8080/video` へ接続
- `yolo26n.pt`を読み込み
- PCの結果画面 `http://127.0.0.1:8765/` をブラウザで開く
- 切断時は1秒間隔で自動再接続

オンラインのAndroidが見つからない場合は、エラーダイアログで知らせます。複数のAndroidがオンラインの場合は誤接続を防ぐため起動を止めるので、使用する1台だけをTailscaleへ接続してください。

## android-camera-streamerの入力仕様

Python側はandroid-camera-streamerのHTTP MJPEG映像を受信します。

```text
http://100.x.x.x:8080/video
```

WebSocket、RTSP、ほかのカメラアプリは使用しません。

## 測定できる内容

- bboxの幅・高さ（入力画像上のpx）
- bbox面積と画面全体に対する面積比
- confidence
- 連続検出フレーム数
- セッション中に安定検出できた最小bbox
- YOLO入力サイズと推論時間
- 任意入力の実物幅とカメラ距離

黄色bboxは未確定、緑bboxは既定3フレーム連続で検出された物体です。単発の誤検出を最小記録には採用しません。

結果は以下へ保存されます。

```text
outputs/yolo26_small_object_lab/<開始日時>/
├── detections.csv
├── measurements.csv
├── auto_*.jpg
└── snapshot_*.jpg
```

推論開始後は操作しなくても、15秒ごとに次の内容を自動保存します。

- その時点のbbox付き画像 `auto_*.jpg`
- `measurements.csv`への測定行
- `stable`（安定検出）、`unstable`（不安定）、`not_detected`（未検出）の状態

対象物から離れてPCへ手が届かなくても記録は継続します。検出できなかった時点も保存されるため、後から検出限界を確認できます。

## 比較設定

既定値:

- モデル: `yolo26n.pt`
- `imgsz`: `960`
- confidence: `0.20`
- 安定検出: `3`フレーム

画面から `imgsz`、confidence、対象COCOクラスを変更できます。鍵やUSBメモリなどCOCOに存在しない物体は、単に小さいのではなく未学習のため検出できない場合があります。

公式資料:

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Tailscale Android](https://tailscale.com/docs/install/android)
