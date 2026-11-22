# YOLO26 Small Object Lab

## 卒業研究用の記録・比較・学習

通常の4手順で起動し、PCの結果画面の「記録開始」か、android-camera-streamerアプリの「Start recording」を押すと、推論前の枠なし元映像を保存します。「記録終了」（アプリでは「Stop recording」）までが1つの撮影セッションです。同一映像による通常検出・固定倍率SR・補間拡大・動的ルール・学習済み選択器の比較、正解注釈の確認画面、評価、選択モデルの教師あり学習を追加しました。

使い方と研究上の条件は [研究用ワークフロー](docs/RESEARCH_WORKFLOW.md)、指標の定義は [正解注釈と評価](docs/ANNOTATION.md) にまとめています。現時点では実データによる選択モデルの学習・効果検証は未実施です。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research --help
```

元映像は1セッション10 GiBまで記録し、欠落・容量到達・書込失敗を画面に表示します。「記録終了」を押すと記録を確定します。記録中にPythonを終了する場合はCtrl+Cで通常終了してください（ウィンドウを閉じると未完了になります）。過去の枠付き保存画像から元動画は復元できません。

android-camera-streamerアプリは、`/video` に接続してきたPCのアドレスへ `http://<PCのTailscale IP>:8765/api/recording/start`（終了は `/stop`）を送ります。アプリ側の設定は不要で、PCが映像を受信し始めるとボタンが有効になり、PC側の記録状態（保存枚数・欠落枚数）も表示します。アプリを使えない場合は、スマホのブラウザで `http://<PCのTailscale IP>:8765/remote` を開いても同じ操作ができます。

PCが枠を描いた検出結果とPCの操作画面（`http://127.0.0.1:8765/` と同じGUI）も、スマホから使えます。アプリ下部のタブで「PC result」（結果映像）、「PC GUI」（操作画面をアプリ内で表示。推論設定・記録・最小記録も操作可）、「Camera」（プレビュー）を切り替えます。どのタブでもカメラ配信と記録は止まりません。ブラウザでは `http://<PCのTailscale IP>:8765/`（GUI）か `/remote`（記録操作と「検出結果を表示」）を開きます。スマホからの映像は通信量を抑えるため最大10fps（`/api/stream?fps=10`）で、表示していない間は受信しません。

対象クラスの初期値は `cup` です（GUIの対象クラス欄を空欄にすると全クラス）。

PC側の待ち受けは `127.0.0.1` とPCのTailscale IPだけで、同じLANの他の端末には公開しません。初回はWindowsのファイアウォール確認が出ることがあるので、プライベートネットワークで許可してください。

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
- `yolo26n.pt`を読み込み（`--model` で変更可）
- PCの結果画面 `http://127.0.0.1:8765/` をブラウザで開く
- 切断時は1秒間隔で自動再接続

ライブ画面で鍵の検出器（Objects365・LeftInCARと拡大画像で学習したもの）を使う場合は、重みを指定して起動します。

```powershell
python .\yolo26_small_object_lab.py --model outputs\key_detector\key_o365_leftincar_all_zoom_v1\weights\last.pt
```

オンラインのAndroidが見つからない場合は、エラーダイアログで知らせます。複数のAndroidがオンラインの場合は誤接続を防ぐため起動を止めるので、使用する1台だけをTailscaleへ接続してください。

## 別のPCで使う

PCがAndroidの映像を取りに行く方式なので、Android側の設定は変えずに別のPCから使えます。

### 準備（初回だけ）

```powershell
git clone https://github.com/DaichiHiraoka/YOLO26SmallObjectLab.git
cd YOLO26SmallObjectLab
uv sync
```

- PCにTailscaleを入れ、Androidと**同じアカウント**でログインする。起動時に `tailscale status --json` でAndroidを探すので、`tailscale` コマンドが使える状態にする（通常のインストールで使えます）。
- 初回起動時にWindowsのファイアウォール確認が出たら、プライベートネットワークで許可する。拒否すると映像は届きますが、アプリの「Start recording」が効きません（PCの結果画面の「記録開始」は使えます）。
- 超解像モデル（FSRCNN）と `yolo26n.pt` は初回実行時に自動で取得します。正解の下書き作成（`propose`）で使う `yolo26x.pt` と `yolo26l.pt` も初回に自動でダウンロードします（合わせて数百MB）。

### 注意

- **同時に使うPCは1台だけ**にする。アプリの記録ボタンは、映像を取りに来たPCへ記録開始を送るため、2台で同時に起動するとどちらに届くか分からなくなります。
- Tailscaleでオンラインにする**Androidも1台だけ**にする。
- PyTorchはCUDA 13版を指定しています。NVIDIA GPUがない、またはドライバーが古いPCではCPUで動きますが、速度は大きく落ちます。処理時間の比較はPCごとに分けて扱ってください。
- 記録は80秒で約0.7GBです。保存先はそのPCのリポジトリ内の `outputs/` です（Gitには含まれません）。

### 動作確認

1. 上の「起動手順」どおりに起動し、PCの結果画面に映像が出て、アプリの記録ボタンが押せることを確かめる。
2. 短く1本記録し、[研究用ワークフロー](docs/RESEARCH_WORKFLOW.md) の `replay` と `evaluate` まで通ることを確かめる。

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
├── recordings/<記録ID>/   （記録開始〜終了ごとの枠なし元映像）
├── detections.csv
├── frames.csv
├── measurements.csv
├── auto_*.jpg
└── snapshot_*.jpg
```

推論開始後は操作しなくても、15秒ごとに次の内容を自動保存します。

- その時点のbbox付き画像 `auto_*.jpg`
- `measurements.csv`への測定行
- `stable`（安定検出）、`unstable`（不安定）、`not_detected`（未検出）の状態

対象物から離れてPCへ手が届かなくても記録は継続します。検出できなかった時点も保存されるため、後から検出限界を確認できます。

## 不安定時の局所超解像

通常は720×720の全フレームをそのままYOLO26へ渡します。次の兆候が出た場合だけ、検出中でbbox面積が最大の物体を選び、bboxの各辺へ20%の余白を加えた局所領域をFSRCNNで超解像して再推論します。切り出しは原則としてbboxの約1.4倍に留まり、bbox自体が画面の大部分を占めない限り全画面にはなりません。

- confidenceが既定`0.45`以下
- bbox面積が画面の既定`1.0%`以下
- 前フレームからconfidenceまたはbbox面積が急低下
- 検出が消えてから2フレーム以内

倍率は対象bboxの短辺を既定96px以上にすることを目標に、`x2`、`x4`、`x8`から動的に選びます。最初の倍率で対象を再検出できない、または通常検出のconfidenceを上回らない場合は、局所画像がYOLO入力上限を大きく超えない範囲で次の倍率へ段階的に上げます。`x4`と`x8`はFSRCNN x2モデルをそれぞれ2回、3回適用するカスケード方式です。

青枠は超解像した領域、マゼンタのbboxと`SR x2`〜`SR x8`表示は局所再推論結果が通常結果より有効で採用されたことを示します。画面から超解像の有効・無効、開始しきい値、目標短辺、最大倍率を変更できます。`detections.csv`と`measurements.csv`には検出元と採用倍率を記録します。

`frames.csv`には検出の有無にかかわらず全処理フレームを記録します。選択倍率、試行した倍率列、発火理由、ROI位置と面積、採用成否、超解像時間、全推論時間が入り、予備分析に利用できます。最適倍率の教師データには、研究用ワークフローの候補別比較ログと人が確認した正解注釈を使います。

カメラ映像と保存画像には、そのフレームでbbox面積が最大の検出結果だけを表示します。内部の安定判定とCSVには、比較・分析用として他の検出結果も引き続き記録します。

手動・自動の測定行は、画像に表示した最大bboxと同じ対象を記録します。現在未検出なら過去の最小bboxを現在値として記録しません。セッション最小の参考表示は別に維持します。

対象選択は[small_object_lab/super_resolution.py](./small_object_lab/super_resolution.py)の`select_super_resolution_target()`へ分離してあります。現在は最大bboxですが、後から別の選択方法へ差し替えられます。

FSRCNN x2モデル（約40KB）は初回使用時に自動取得し、`models/FSRCNN_x2.pb`へキャッシュします。取得後はSHA-256を検証します。モデルはOpenCVのGSoC成果として公開されている[Saafke/FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow)（Apache-2.0）を使用します。

## 比較設定

既定値:

- モデル: `yolo26n.pt`
- `imgsz`: `960`
- confidence: `0.20`
- 安定検出: `3`フレーム
- 局所超解像: `FSRCNN x2 / x4 / x8`
- 超解像後の目標短辺: `96px`
- 最大倍率: `x8`

画面から `imgsz`、confidence、対象COCOクラスを変更できます。鍵やUSBメモリなどCOCOに存在しない物体は、単に小さいのではなく未学習のため検出できない場合があります。

公式資料:

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [Tailscale Android](https://tailscale.com/docs/install/android)
