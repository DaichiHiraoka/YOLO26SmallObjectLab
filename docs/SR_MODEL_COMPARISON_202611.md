# FSRCNN以外の超解像モデルの比較（2026年11月）

[作業指示](TASK_SR_MODEL_COMPARISON.md) に従い、OpenCVで使える超解像モデル（ESPCN・LapSRN・EDSR）を、現行のFSRCNN・bicubic・切り出しのみと、撮影済みの鍵の記録で比べた。目的は、低スペックのデバイスへの組み込みを前提に、**処理時間に見合う精度の上積みがあるか**を確かめることである（[10月のレポート](REPORT_202610.md) 7章）。

## 1. 条件

### 入力

| 入力 | パス | 選んだ理由 |
|---|---|---|
| 鍵の記録 | `outputs/yolo26_small_object_lab/20261027_093747_754671/recordings/20261027_094006_823995` | 10月の遠距離の鍵の映像（754フレーム、43.9秒、720×720、欠落0） |
| `predictions.csv` | `outputs/research/key_far_20261027_1280_fix1/predictions.csv` | 上の記録を最終の鍵検出器（`--label key`、全画面1280px）で比較実行した結果。採用ルールの修正後のもので、10月の `probe_enlargement.py`（`outputs/research/probe_enlargement_20261027.json`）でも使った |
| 最終の鍵検出器 | `outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt` | READMEに記載の最終モデル |
| 開発用の鍵検出器 | `outputs/key_detector/key_o365_leftincar_bydate_zoom_v1/weights/best.pt` | 10月の調査で「開発用モデル」として比べたもの（撮影日で分けたデータ＋拡大画像で学習）。FSRCNNで信頼度が上がった検出器 |

### 実行したコマンド

```powershell
# 1回目：指示書の既定（ESPCN・LapSRN、x2/x4）
.\.venv\Scripts\python.exe scripts\compare_sr_models.py outputs/yolo26_small_object_lab/20261027_093747_754671/recordings/20261027_094006_823995 outputs/research/key_far_20261027_1280_fix1/predictions.csv --weights outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt outputs/key_detector/key_o365_leftincar_bydate_zoom_v1/weights/best.pt --threads 4 --output outputs/research/sr_models_01.json

# 2回目：x8とEDSR（精度の上限の参考）を追加
.\.venv\Scripts\python.exe scripts\compare_sr_models.py <同じ記録> <同じpredictions.csv> --weights <同じ2つ> --models espcn lapsrn edsr --scales 2 4 8 --threads 4 --output outputs/research/sr_models_02_x8_edsr.json
```

標本の既定は `probe_enlargement.py` と同じ（`--min-time 24`、`--factors 0.25 1/6`、`--limit 80`、余白100%、検出器の入力640px）。

| グループ | 内容 | 標本数 | 切り出しの長辺（中央値） |
|---|---|---|---|
| 実映像 | 24秒以降で、baselineが信頼度0.6以上で検出した遠い鍵 | 22 | 88px |
| 1/4縮小 | 9〜30秒の鮮明なフレームを1/4に縮小し、JPEG品質75で圧縮 | 80 | 69px |
| 1/6縮小 | 同じく1/6 | 80 | 45px |

### 環境
- CPU：AMD Ryzen 7 7700（8コア16スレッド）、GPU：NVIDIA GeForce RTX 5070
- OpenCV 5.0.0、Ultralytics 8.4.86、torch 2.13.0+cu130
- 拡大処理はCPU（OpenCV dnn）、`--threads 4`。検出（YOLO）はGPU。
- 超解像モデルは初回に `models/` へ取得し、SHA-256を検証した（ESPCN x2/x4、LapSRN x2/x4/x8、EDSR x2/x4）。ESPCNとEDSRのx8は、x4とx2の組み合わせ、FSRCNNのx4/x8はx2の繰り返しである。

## 2. 結果

再検出できた数（重なりIoU 0.3以上・信頼度0.25以上）と、平均信頼度（括弧内）。各グループの最良の数を太字にした。

### 2.1 最終の鍵検出器（`key_o365_leftincar_all_zoom_v1`）

| 方法 | 実映像（22） | 1/4縮小（80） | 1/6縮小（80） |
|---|---|---|---|
| 切り出しのみ | **22**（0.86） | 77（0.85） | **75**（0.80） |
| bicubic x2 | 21（0.75） | 79（0.88） | 74（0.79） |
| bicubic x4 | 20（0.74） | **80**（0.89） | 73（0.80） |
| bicubic x8 | 20（0.71） | 78（0.88） | 74（0.80） |
| FSRCNN x2（現行） | 19（0.69） | 74（0.81） | 69（0.75） |
| FSRCNN x4（現行） | 18（0.66） | 73（0.75） | 66（0.69） |
| FSRCNN x8（現行） | 19（0.64） | 70（0.74） | 66（0.68） |
| ESPCN x2 | 19（0.66） | 79（0.86） | 72（0.77） |
| ESPCN x4 | 12（0.42） | 63（0.66） | 54（0.57） |
| ESPCN x8 | 12（0.42） | 62（0.65） | 54（0.56） |
| LapSRN x2 | 21（0.79） | 74（0.84） | 71（0.77） |
| LapSRN x4 | 14（0.50） | 63（0.64） | 46（0.47） |
| LapSRN x8 | 8（0.30） | 56（0.56） | 45（0.41） |
| EDSR x2 | 20（0.66） | 75（0.76） | 64（0.64） |
| EDSR x4 | 15（0.54） | 58（0.55） | 37（0.37） |
| EDSR x8 | 15（0.53） | 56（0.53） | 34（0.34） |

### 2.2 開発用の鍵検出器（`key_o365_leftincar_bydate_zoom_v1`）

| 方法 | 実映像（22） | 1/4縮小（80） | 1/6縮小（80） |
|---|---|---|---|
| 切り出しのみ | 20（0.64） | 70（0.64） | 70（0.61） |
| bicubic x2 | 19（0.56） | 72（0.60） | **73**（0.59） |
| bicubic x4 | 19（0.57） | 77（0.61） | 68（0.59） |
| bicubic x8 | 17（0.56） | 76（0.61） | 70（0.58） |
| FSRCNN x2（現行） | 20（0.65） | 69（0.56） | 66（0.56） |
| FSRCNN x4（現行） | 20（0.63） | 69（0.55） | 65（0.55） |
| FSRCNN x8（現行） | 20（0.62） | 64（0.52） | 66（0.56） |
| ESPCN x2 | 20（0.63） | 69（0.57） | 70（0.60） |
| ESPCN x4 | 15（0.47） | 60（0.50） | 72（0.62） |
| ESPCN x8 | 15（0.45） | 59（0.49） | **73**（0.63） |
| LapSRN x2 | **21**（0.64） | **78**（0.64） | 70（0.61） |
| LapSRN x4 | 16（0.51） | 55（0.44） | 68（0.56） |
| LapSRN x8 | 17（0.47） | 53（0.43） | 72（0.57） |
| EDSR x2 | 20（0.63） | 67（0.53） | 64（0.53） |
| EDSR x4 | 20（0.59） | 55（0.43） | 57（0.45） |
| EDSR x8 | 19（0.59） | 56（0.42） | 57（0.43） |

### 2.3 拡大処理の時間（中央値、CPU 4スレッド）

| 方法 | 実映像（切り出し88px） | 1/4縮小（69px） | 1/6縮小（45px） |
|---|---|---|---|
| 切り出しのみ | 0ms | 0ms | 0ms |
| bicubic x2 / x4 / x8 | 0.1ms | 0.1ms | 0.1ms |
| FSRCNN x2 / x4 / x8 | 5.2 / 10.1 / 48.0ms | 5.2 / 9.8 / 46.5ms | 2.8 / 4.6 / 20.4ms |
| ESPCN x2 / x4 / x8 | 3.5 / 2.4 / 30.2ms | 3.4 / 2.1 / 28.9ms | 1.4 / 0.8 / 12.0ms |
| LapSRN x2 / x4 / x8 | 15.0 / 70.1 / 342ms | 13.5 / 64.6 / 329ms | 5.3 / 28.2 / 134ms |
| EDSR x2 / x4 / x8 | 834 / 829 / 17,020ms | 782 / 764 / 16,337ms | 394 / 356 / 7,860ms |

- 2回目の実行の値。1回目（ESPCN・LapSRN、x2/x4のみ）と比べ、時間は実行ごとに数ms程度ぶれた（例：FSRCNN x2は1回目2.9ms、2回目5.2ms。1/4縮小）。精度は2回とも同じだった。
- ここに検出（YOLO、GPU）の時間は含まない。低スペックのデバイスでは、検出自体の時間も大きく増える。

## 3. 結論

1. **ESPCN・LapSRN・EDSRに、処理時間に見合う上積みはない。**
   - 最終の検出器では、どのグループでも最良は切り出しのみかbicubicだった。超解像モデルはどれもこれを上回らない。
   - 倍率x4・x8にすると、ESPCN・LapSRN・EDSRは大きく下がった（例：最終の検出器・1/6縮小で、切り出しのみ75に対し、ESPCN x4は54、LapSRN x4は46、EDSR x4は37）。
   - EDSRは1回0.4〜0.8秒（x8は8〜17秒）かかり、低スペックのデバイスでは使えない。精度の上限の参考としても、上積みは見られなかった。
2. **2つの検出器で順位が一致しない。**
   - 開発用の検出器では、LapSRN x2が実映像（21/22）と1/4縮小（78/80）で最良だった。しかし最終の検出器では、LapSRN x2は切り出しのみより低い（22→21、77→74）。
   - 開発用の検出器の1/6縮小では、ESPCN x8（73）がbicubic x2（73）と並んだが、最終の検出器では54と大きく下がった。
   - よって、**超解像モデルの「効く・効かない」は、検出器1つに依存した結果として扱う**。10月の調査（[REPORT_202610.md](REPORT_202610.md)）で見た、FSRCNNで方向が逆転する現象と同じである。
3. **処理時間と精度のバランスでは、切り出しのみ（拡大処理0ms）とbicubic（0.1ms）が最も良い。** 超解像を使う理由は、今回の条件では見つからなかった。最も軽いESPCN x2（1.4〜3.5ms）でも、両検出器でbicubicを一貫して上回ってはいない。

## 4. 注意
- **正解は人が確認した正解枠ではない。** 比較実行のbaselineの検出（信頼度0.6以上）を正解とみなしており、縮小グループではその枠を同じ比率で縮めている。baselineが見つけられた鍵だけが標本になるので、見つけやすい鍵に偏っている。
- **鍵の検出器は、bicubicとFSRCNNで拡大した画像を混ぜて学習している。** ESPCN・LapSRN・EDSRで拡大した画像は見慣れない画質になるので、これらのモデルに不利である。公平に比べるには、拡大方法を均等に含めて学習した検出器（できれば乱数シード違いで複数）が必要である。
- 実映像の標本は22個と少なく、1個の差が約5ポイントに相当する。
- 縮小グループは人工的な縮小（INTER_AREA＋JPEG品質75）で、実際に遠くから撮った映像のボケや雑音は再現していない。
- 時間は開発用PCのCPU（4スレッドに制限）での値で、組み込み先の機種での値ではない。

## 5. 出力
- `outputs/research/sr_models_01.json`（1回目）
- `outputs/research/sr_models_02_x8_edsr.json`、`outputs/research/sr_models_02_x8_edsr.log`（2回目）
