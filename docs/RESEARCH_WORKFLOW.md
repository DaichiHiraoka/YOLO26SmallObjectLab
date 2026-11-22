# 卒業研究用の実験手順

この実装は「小物体の周辺切り出し範囲と超解像倍率を選ぶモデル」の研究基盤です。元映像保存、比較実行、正解注釈、教師あり学習まで接続しています。本人が撮影・確認したデータによる学習と有効性の検証は、まだ実施していません。

検証は2段階で進めます。まずカップとCOCO学習済みの `yolo26n.pt`（どちらも既定値）で、撮影から評価までを最後まで通して仮説を確かめます。その後、鍵と自作の鍵検出器で本命の検証を行います（`--model` と `--label key` を指定。詳細は [研究計画](RESEARCH_PLAN_20260929.md)）。以下のコマンド例は、特記がない限りカップの手順です。

## 1. いつもの4手順で撮影

android-camera-streamerを起動し、Tailscale IPを選び、PCのTailscaleと `yolo26_small_object_lab.py` を起動します。起動しただけでは元映像を保存しません。PCの結果画面の「記録開始」、またはandroid-camera-streamerアプリの「Start recording」を押すと、入力映像を推論・枠描画の前に保存します。「記録終了」（アプリでは「Stop recording」）を押すと書込待ちを完了して記録を確定します。アプリが使えないときは、スマホのブラウザで `http://<PCのTailscale IP>:8765/remote` を開いても操作できます。開始から終了までが1つの撮影セッション（`session_id` ＝ 記録ID）で、Pythonを起動し直さずに何本でも撮れます。

結果画面とアプリに、記録状態・保存枚数・欠落枚数・容量を表示します。記録中にPythonを終了する場合はCtrl+Cで通常終了すれば記録を確定します。ウィンドウを閉じる・強制終了した記録は未完了として扱います。ブラウザを閉じるだけではPythonは終了しません。

```
outputs/yolo26_small_object_lab/<日時>/
  recordings/<記録ID>/
    manifest.json           記録状態、設定変更履歴、欠落、入力寸法
    frames.csv              frame_id、PC受信相対時刻、画像パス
    frames/00000001.png     枠のない入力画像
  frames.csv                推論ログ（recording_idとraw_frame_idで元画像と対応）
  detections.csv
  measurements.csv
  auto_*.jpg
```

PNGはデコード後の画素を保持します。スマホが行ったJPEG圧縮を元に戻すものではありません。時刻はPC受信時刻であり、カメラ露光時刻ではありません。推論が処理しなかった受信フレームも記録対象です。書込キューは64枚、画像容量は1セッション10 GiBが上限です。書込遅延・容量到達・書込失敗は画面とmanifestへ明記します。欠落のある記録は通常の比較実行では拒否します。

最初は、対象クラスをcup、同時に映る対象を1個にした短い動画から始めてください。カメラを固定し、物体を近距離から遠距離へ移す・横に動かす・一部隠す条件を別の撮影にします。同一物体、似た背景、同じ撮影日の映像を学習用と最終評価用にまたがって配置しないよう、撮影セッションをまとめて割り当てます。

## 2. 同じ映像で比較

以下はリポジトリのフォルダで実行します。`<日時>`、`<記録ID>`、出力名は実際の記録に置き換えます。出力フォルダは新しい名前を指定し、以前の結果を上書きしません。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research replay outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> --output outputs/research/run_01
```

対象を鍵にする場合は、鍵の検出器の重みとクラス名を指定します（既定は `yolo26n.pt` と `cup`）。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research replay outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> --output outputs/research/run_01 --model outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt --label key
```

既定では次の4方式を実行します。

| 名前 | 処理 |
|---|---|
| baseline | 通常のYOLO検出 |
| fixed | 不安定時に固定の余白20%・SR x4 |
| bicubic | fixedと同じ範囲・倍率、超解像を通常補間に置換 |
| rule | 既存の短辺96pxを目指すx2/x4/x8規則、改善しなければ上位倍率を試す |
| learned | 学習済み選択器。モデルを指定した場合だけ実行可能 |

全方式で同じ元フレーム・信頼度閾値（既定0.20）・同じ単一対象追跡処理を使います。最初は最大bboxから開始し、追跡中は面積が大きい別物体へ毎フレーム選び直しません。直近位置と速度で最大0.75秒まで回復領域を予測し、SRで実際に再検出した結果で状態を更新します。予測位置だけを検出成功として数えません。

これは研究用の簡易な単一対象追跡であり、ByteTrackや汎用多物体追跡ではありません。実験中に別の同クラス物体を入れないことが最小構成の前提です。ライブ表示は従来の最大bbox方式を維持し、この共通追跡処理はオフライン比較側で使用します。

全画面YOLOはフレームごとに一度実行し、その同じ結果を各方式に渡します。方式ごとの時間は、その通常推論時間と各方式自身の追加処理時間の合計です。モデルの初期化・ウォームアップ・ディスクI/O・ネットワークは除外します。`compute_fps`はこの処理時間から算出した値で、ライブ配信FPSとは区別してください。各方式は並列に走るのではなく順次測定します。

局所入力は既定640角、全画面入力は960角です。全方式で固定し、`rect=False`で前処理条件をそろえます。局所画像をYOLOが再拡大するので、SR x2とx8でYOLO入力上のbboxサイズが同じになることがあります。実際の入力上の対象サイズと補間倍率は `attempts.jsonl` に記録します。SR倍率をそのまま検出器入力上の拡大率と解釈しないでください。

出力: `run.json`（設定、重み/コード/入力のハッシュ、実行完了状態、処理時間）、`predictions.csv`（未検出を含む全フレーム）、`attempts.jsonl`（処理領域、倍率、各試行結果と時間）。古い枠付きJPEGは正式な元映像の代わりに使えません。

## 3. 正解を確認

まず、比較対象より大きい検出器（既定は `yolo26x.pt` と `yolo26l.pt`）で、正解枠の下書きを自動で作ります。全画面で見つからないフレームは直前の位置の周りを切り出して探し（超解像は使わない）、ほぼ動いていない2つの下書きの間の短い空白（15フレームまで）は補間します。下書きは正解ではありません。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research propose outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> --output outputs/annotations/session_01.proposals.csv --label cup
.\.venv\Scripts\python.exe -m small_object_lab.research annotate outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> --output outputs/annotations/session_01.csv --label cup --proposals outputs/annotations/session_01.proposals.csv --stride 3
```

`--proposals` を付けると、最初に下書きを12枚ずつ並べた画面が開きます。枠がずれている・大きすぎる・別の物を囲んでいるタイルをクリックして赤い×を付け、Space（またはEnter）でそのページの残りを確定します。Pで前のページ、Qで終了（途中から再開可）。×を付けたフレームと下書きのないフレームは、次に画面全体を8枚ずつ並べた画面に出ます。対象が少しでも見えるフレームだけをクリックしてオレンジ枠を付け、Spaceで残りを「対象なし」として確定します。オレンジ枠のフレームは、最後に1枚ずつの画面で手で囲みます。どの方法で確定したか（下書きをそのまま採用・1枚ずつの画面で確認・手で描画・一覧で不在を確認・1枚ずつ不可視）は `session_01.csv.review.json` に記録されるので、卒論では「下書きのうち何%をそのまま採用したか」を報告できます。

下書きのないフレームは、大きい検出器でも見つけにくい、評価で重要なフレームです。省略すると評価が簡単なフレームに偏るので、`--stride` は下書きの有無に関係なく同じ間隔で適用します。鍵の場合、COCOの検出器には鍵のクラスがないため `--models` に鍵の検出器を指定しますが、比較に使う検出器と同じになる点を結果の考察で明記してください。

操作: Rで枠を描く、Sで確認保存して次へ、Iで完全不可視/不在を確認して次へ、Nで未確認のまま次へ（保存されず、採点にも使われません）、Pで戻る、Cで直前の枠を下書きとしてコピー、Uで未確認に戻す、Qで終了。Rの枠選択はEnter/Spaceで確定、Escで取消。保存後は途中から再開できます。枠のコピーだけでは確認済みになりません。画面には英字で操作を表示します。

`--start-frame`・`--end-frame` で確認するフレームの範囲を、`--stride 3` のように間引き（3フレームに1枚）を指定できます。範囲外・間引いたフレームは未確認のまま残り、採点されません。鍵の場合は、鍵・リング・カラビナ・キーホルダーをまとめた鍵束全体を1つの枠で囲みます。

部分遮蔽で対象を同定できる場合は可視部分を囲み、完全に隠れたフレームはIにします。正解の規則を途中で変えず、最終評価前に指導教員と確認してください。未知のフレームを不在扱いにしません。

CSVだけの雛形が必要なら `annotate` を `annotation-template` に置き換えます。雛形は全行 `reviewed=0` です。指標の定義は [ANNOTATION.md](ANNOTATION.md) を参照してください。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research evaluate outputs/research/run_01/predictions.csv outputs/annotations/session_01.csv --output outputs/research/evaluation_01.json
```

可視で確認済みの正解に対し、同じクラスかつIoU 0.5以上を正検出として採点します。未確認しかなければ精度は `null` です。IDの変化、見失い、回復までの時間、サイズ別recallも出力します。これはmAP、HOTA、IDF1の計算ではありません。

## 4. 候補の採点と学習

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research replay outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> --output outputs/research/candidates_01 --methods baseline --collect-candidates
```

`candidates.jsonl` に、各辺余白10%/20%/50%/100%/200%/300%とSR x2/x4/x8の組（最大18組）、同じ6段階の余白で切り出すだけ（x1、超解像なし）の6組、追加処理なしを記録します。余白は各辺に適用し、最低8pxです。拡大後の局所画像が640pxを超えるSR候補は省きます。x1はYOLO自身の縮小・拡大だけなので常に候補に入ります。鍵のような小物は周囲が写らないと判別できないため、広い余白も候補にしています。利用できない候補を失敗扱いして学習しません。候補収集は通常検出だけで更新した共通の履歴を使います。そのため、学習後は必ず `learned` で動画全体を再生し、モデル自身の判断が次フレームに及ぼす影響を含めて評価します。

複数の撮影セッションをまとめる場合:

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research assemble --runs outputs/research/candidates_01 outputs/research/candidates_02 --annotations outputs/annotations/session_01.csv outputs/annotations/session_02.csv --output outputs/dataset_v1
```

`sessions.json` を見て撮影セッションを割り当て、`split.json` を作ります。以下のIDは例示用で、そのままでは実データと一致しません。動画内の隣接フレームで学習/評価を分けません。

```json
{"train":["撮影セッションA"],"validation":["撮影セッションB"],"test":["撮影セッションC"]}
```

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research train-selector outputs/dataset_v1/samples.jsonl outputs/dataset_v1/annotations.csv outputs/dataset_v1/split.json --output outputs/models/selector_v1.json
.\.venv\Scripts\python.exe -m small_object_lab.research replay outputs/yolo26_small_object_lab/<評価用の日時>/recordings/<記録ID> --output outputs/research/test_v1 --methods baseline fixed bicubic rule learned --selector outputs/models/selector_v1.json
```

学習器は、bbox比率・信頼度・正規化移動速度・欠落時間・面積と信頼度の変化から、候補ごとの成績を予測する軽量ridge回帰です。報酬は正解との一致（1または0）から追加処理時間×既定0.002を引きます。閾値やペナルティは調整用データで決め、最終評価前に固定します。信頼度の高さ自体を正解にはしません。

学習に使うのは、実行時に学習器が実際に選ぶフレーム（対象があり、局所処理の候補があり、固定・ルール方式と共通の拡大条件 `needs_enhancement` が成立するフレーム）だけです（`--frames decision`、既定）。対象のいないフレームまで含めると「何もしない」が過大評価され、拡大を控えすぎて追跡が途切れます。通常検出が今のフレームで何も見つけていないとき、「何もしない」の成績は推定せず0とします。`--ridge-alpha` は行動ごとの推定の縮小の強さで、候補が24通りあり学習フレームが数百の規模では、既定の1だと過学習します。カップの調整用データでは `--ridge-alpha 1000 --latency-penalty 0.0005` を選びました。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research train-selector outputs/dataset_v1/samples.jsonl outputs/dataset_v1/annotations.csv outputs/dataset_v1/split.json --ridge-alpha 1000 --latency-penalty 0.0005 --output outputs/models/selector_v1.json
```

学習と標準化はtrainだけ、モデル内の参考成績はvalidationだけで計算します。testは学習・調整に使用しません。学習に使った同じsession_idの通常評価は拒否します。モデル出力は透明なJSONで、重みや学習セッション、分割、学習入力のハッシュを残します。モデルが学習していない候補は選びません。

## 5. FSRCNN以外の超解像モデルを試す

[10月のレポート](REPORT_202610.md) 7.2節の候補のうち、OpenCVで読めるESPCN・LapSRN・EDSRを、撮影済みの記録で比べます。モデルは初回に `models/` へ自動で取得し、SHA-256を検証します。SPAN・RLFNなどは未対応です。

```powershell
.\.venv\Scripts\python.exe scripts\compare_sr_models.py outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID> outputs/research/<比較実行>/predictions.csv --weights outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt --threads 4 --output outputs/research/sr_models_01.json
```

- 標本は `scripts/probe_enlargement.py` と同じです（遠い実映像の鍵と、鮮明なフレームを縮小した鍵。正解は比較実行のbaselineの検出で、人が確認した正解枠ではありません）。
- 切り出しのみ・bicubic・FSRCNN（現行）・指定したモデルを、同じ切り出し（余白100%）で比べます。既定はESPCNとLapSRN、倍率はx2とx4です。EDSRは非常に重いので、上限の参考にするときだけ `--models espcn lapsrn edsr` で加えます。
- 表示する時間は拡大処理だけの中央値で、CPU（OpenCV）で計測します。`--threads` でスレッド数を制限できます。
- 鍵の検出器はbicubicとFSRCNNの拡大画像を混ぜて学習しているので、ほかのモデルは不利です。`--weights` に別の検出器を並べるか、COCOのモデルを `--label` 付きで指定して、順位が検出器1つに依存していないかを確かめてください。
- 人が確認した正解枠があれば、`predictions.csv` の代わりに `--annotations <注釈CSV>` を指定すると、正解枠から標本を作ります（baselineが見落とした対象も含まれます。縮小は `--factors` で指定）。2026年11月の比較では、COCOの `yolo26n.pt`・`yolo26s.pt` と `--label cup` で、カップの正解枠を使いました。

比較の結果、比較実行とライブ画面の超解像の既定はESPCNにしました（[超解像モデルの比較](SR_MODEL_COMPARISON_202611.md)）。`replay` の `--sr-model fsrcnn` で、2026年11月以前と同じFSRCNNの結果を再現できます。それ以前に学習した選択モデルはFSRCNNの候補で学習しているので、`--sr-model fsrcnn` のときだけ使えます。

## 現時点で本人の作業を待つ部分

- 新しい枠なし元映像の収集と正解注釈の確認。
- 研究対象・実験条件・成功基準の指導教員との合意。
- 実データでの学習、未使用映像での効果検証、結果の最終解釈。

テストで使用する合成データや過去の枠付き画像を、研究結果や学習済み成果として扱わないでください。候補収集・固定倍率・ルール方式・学習方式を同一の設定で比較し、同じデータの内容が別session_idで重複しないよう撮影単位で管理します。
