# FSRCNN以外の超解像モデルの比較（2026年11月）

[作業指示](TASK_SR_MODEL_COMPARISON.md) に従い、OpenCVで使える超解像モデル（ESPCN・LapSRN・EDSR）を、現行のFSRCNN・bicubic・切り出しのみと、撮影済みの鍵の記録で比べた。目的は、低スペックのデバイスへの組み込みを前提に、**処理時間に見合う精度の上積みがあるか**を確かめることである（[10月のレポート](REPORT_202610.md) 7章）。

> **追記（結論の更新）**：1〜5節の鍵での比較では超解像の効果が見えなかったが、これは自作の鍵検出器の偏り（bicubicとFSRCNNの拡大画像で学習、わずかな変化に不安定）と、baselineを正解とした標本の偏りによるものと判断した。人の正解枠があるカップの記録と、拡大方法に偏りのないCOCOの汎用モデルで比べ直すと（[6節](#6-カップでの比較人の正解枠coco汎用モデル)）、**超解像は2つの検出器のどちらでも、切り出しのみ・bicubicの1.5〜4倍の対象を再検出した**。処理時間とのバランスが最も良い **ESPCN を既定の超解像モデルにした**（7節）。

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

## 6. カップでの比較（人の正解枠、COCO汎用モデル）

鍵での比較は検出器の偏りが強く、超解像モデルの比較に向いていなかった（4節の注意）。そこで、次の条件で比べ直した。

### 条件
- **記録**：2026-11-17のカップの記録4本（`outputs/yolo26_small_object_lab/20261117_020819_745230/recordings/` の `20261117_021645_944529`、`021829_946788`、`022205_950242`、`022355_294680`。同じ撮影条件）
- **正解**：人が確認した正解枠（`outputs/annotations/cup_<記録ID>.csv` の `reviewed=1, visible=1`）。baselineが見落としたカップも標本に含まれる。1本あたり60フレームを記録全体から均等に選んだ。
- **検出器**：COCOの `yolo26n.pt` と `yolo26s.pt`（`--label cup`）。超解像の画像で追加学習しておらず、拡大方法に偏りがない。
- **標本のグループ**：実映像のまま（カップの長辺は約30〜70px、中央値41〜48px）と、1/2・1/3・1/4に縮小しJPEG品質75で圧縮したもの。
- **切り出し**：余白100%と200%の2通り。
- 比較スクリプトに、人の正解枠から標本を作る `--annotations` を追加して実行した。

```powershell
.\.venv\Scripts\python.exe scripts\compare_sr_models.py outputs/yolo26_small_object_lab/20261117_020819_745230/recordings/<記録ID> --annotations outputs/annotations/cup_<記録ID>.csv --weights yolo26n.pt yolo26s.pt --label cup --models espcn lapsrn --scales 2 4 8 --factors 0.5 0.3333 0.25 --limit 60 --threads 4 --output outputs/research/sr_models_cup_<記録ID>.json
# 余白200%は --padding 2.0、出力名 sr_models_cup_pad200_<記録ID>.json
```

### 結果（4本の合計。各グループ240標本、計960）

再検出できた数（IoU 0.3以上・信頼度0.25以上）。時間は1/3縮小のグループの拡大処理の中央値（CPU 4スレッド）。

| 方法 | 時間（余白100% / 200%） | yolo26n 余白100% | yolo26n 余白200% | yolo26s 余白100% | yolo26s 余白200% |
|---|---|---|---|---|---|
| 切り出しのみ | 0ms | 43 | 49 | 72 | 95 |
| bicubic x2 / x4 / x8 | 0.0〜0.1ms | 53 / 47 / 45 | 44 / 41 / 39 | 57 / 67 / 71 | 91 / 89 / 89 |
| FSRCNN x2 | 2.0 / 4.4ms | 125 | 111 | 168 | 144 |
| FSRCNN x4 | 3.3 / 8.3ms | 171 | 122 | 213 | 165 |
| FSRCNN x8 | 13.5 / 37.5ms | 181 | 126 | **226** | 164 |
| ESPCN x2 | 1.1 / 2.8ms | 134 | 103 | 127 | 145 |
| **ESPCN x4** | **0.6 / 1.8ms** | 195 | 124 | 181 | 154 |
| ESPCN x8 | 8.2 / 23.7ms | 194 | 122 | 184 | 159 |
| LapSRN x2 | 3.9 / 11.5ms | 136 | 99 | 158 | **173** |
| LapSRN x4 | 18.9 / 53.3ms | 207 | **171** | 161 | 165 |
| LapSRN x8 | 79.1 / 257.7ms | **253** | 152 | 187 | 140 |

グループ別の内訳（余白100%、yolo26n。実映像 / 1/2 / 1/3 / 1/4）：切り出しのみ 33 / 4 / 6 / 0、bicubic x4 26 / 7 / 11 / 3、FSRCNN x4 54 / 59 / 35 / 23、**ESPCN x4 57 / 66 / 50 / 22**、LapSRN x8 62 / 70 / 78 / 43。

参考：比較実行の通常の検出（全画面960px、`outputs/research/cup_20261117_*`）が人の正解枠と一致した割合は、4本で14〜49%だった。もともと検出の難しい場面である。

### 読み取れること
1. **超解像は、2つの検出器・2つの余白・すべての大きさで、切り出しのみとbicubicを大きく上回った**（合計で1.5〜4倍。差が最も小さいのはyolo26s・余白200%で、切り出しのみ95に対し超解像140〜173）。bicubicは切り出しのみとほぼ同じで、単なる拡大では効かない。
2. どの超解像モデルが最良かは、検出器と余白で入れ替わる（yolo26nではLapSRN・ESPCN、yolo26sではFSRCNN）。どの条件でも上位にあるのはFSRCNN x4とESPCN x4である。
3. 余白は200%より100%のほうが全体に良かった。
4. 鍵での結果（1〜5節）と違うのは、検出器の違いによる。超解像の効果を測るには、拡大方法に偏りのない検出器と、人の正解枠を使う必要がある。

## 7. 既定の超解像モデルをESPCNにした

- **理由**：ESPCN x4は、拡大処理0.6〜1.8ms（FSRCNN x4の約1/5、LapSRNの1/30以下）で、カップの比較では最良の方法の7〜9割、切り出しのみ・bicubicの1.6〜4.5倍の対象を再検出した。低スペックのデバイスを前提にした、処理時間と精度のバランスで最良である。
- **変更点**
  - `small_object_lab/super_resolution.py`：`create_super_resolution(name)` と既定値 `DEFAULT_SR_MODEL = "espcn"` を追加。ESPCNのx8はx4とx2の組み合わせ。
  - 比較実行：`--sr-model {espcn,fsrcnn,lapsrn,edsr}`（既定 `espcn`）。`ExperimentConfig.sr_model` として `run.json` と候補の記録に残る。
  - ライブ画面：起動時の `--sr-model`（既定 `espcn`）。画面と映像の表示に使用中のモデル名を出す。
- **以前の結果との互換**：2026年11月以前の比較実行・候補収集・選択モデル（`selector_cup_*` など）はFSRCNNで測ったもの。`sr_model` が記録されていない古い選択モデルはFSRCNN扱いとし、`--sr-model fsrcnn` のときだけ使える（ESPCNの実行では拒否する）。
- 鍵の検出器の学習用の拡大画像（`scripts/prepare_key_dataset.py`）は、学習データの再現性のため、FSRCNNのままにしている。

### 出力
- `outputs/research/sr_models_cup_<記録ID>.json`（余白100%）、`outputs/research/sr_models_cup_pad200_<記録ID>.json`（余白200%）、各ログ `sr_models_cup_20261117.log`・`sr_models_cup_pad200_20261117.log`
