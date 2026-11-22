# 作業指示：FSRCNN以外の超解像モデルを撮影済みの記録で比較する

PC上のClaude Codeセッション向けの指示です。上から順に実行し、最後に結果の文書を作ってpushしてください。

## 目的

[10月のレポート](REPORT_202610.md) 7.2節の候補のうち、OpenCVで使えるESPCN・LapSRN（必要ならEDSR）が、現行のFSRCNN・bicubic・切り出しのみと比べて、小さい鍵の再検出に効くかを、撮影済みの鍵の記録で確かめる。比較の道具は `scripts/compare_sr_models.py`（使い方は [実験手順](RESEARCH_WORKFLOW.md) 第5節）。

## 1. コードを用意する

- リポジトリ：https://github.com/DaichiHiraoka/YOLO26SmallObjectLab
- ブランチ：`claude/current-situation-fl9j84`

記録・検出器の重み・比較実行の結果は `outputs/` にあり、Gitには入っていない。**まず、このPCの既存のクローン**（`outputs/key_detector` があるフォルダ。`outputs` はEドライブへのリンクの場合がある）を探し、そこで作業する。

```powershell
git status            # 未コミットの変更があれば、消さずに報告して止まる
git fetch origin
git switch claude/current-situation-fl9j84
git pull
uv sync
```

既存のクローンが見つからない場合だけ、新しいフォルダにクローンし、データは既存の `outputs/` を絶対パスで指定する。

```powershell
git clone -b claude/current-situation-fl9j84 https://github.com/DaichiHiraoka/YOLO26SmallObjectLab.git
cd YOLO26SmallObjectLab
uv sync
```

## 2. 入力を探す

| 入力 | 探す場所 | 条件 |
|---|---|---|
| 鍵の記録 | `outputs/yolo26_small_object_lab/<日時>/recordings/<記録ID>/` | 10月の調査で使った**遠距離の鍵の映像**（約754フレーム、30秒以降に鍵が24〜32px）。`manifest.json` の日時と [10月のレポート](REPORT_202610.md) 5.2〜5.5節を手がかりにする |
| `predictions.csv` | `outputs/research/*/predictions.csv` | 上の記録を**鍵の検出器**で比較実行した結果。`method=baseline` の行があり、`label` が `key` |
| 最終の鍵検出器 | `outputs/key_detector/key_o365_leftincar_all_zoom_v1/weights/last.pt` | README記載のもの |
| 開発用の鍵検出器 | `outputs/key_detector/*/weights/` のほかの重み | 10月の調査で「開発用モデル」として比べたもの。分からなければ最も新しい別の学習結果 |

候補が複数あるときは、選んだものと理由を結果の文書に書く。`scripts/probe_enlargement.py` の過去の出力（JSON）が `outputs/` にあれば、そのときの引数が分かるので優先する。

鍵の検出器による `predictions.csv` がない場合は、先に作る（新しい出力フォルダを指定する）。

```powershell
.\.venv\Scripts\python.exe -m small_object_lab.research replay <記録> --output outputs/research/sr_models_baseline --methods baseline --model <最終の鍵検出器> --label key
```

## 3. 実行する

```powershell
.\.venv\Scripts\python.exe scripts\compare_sr_models.py <記録> <predictions.csv> --weights <最終の鍵検出器> <開発用の鍵検出器> --threads 4 --output outputs/research/sr_models_01.json
```

- ESPCN・LapSRNのモデルは初回に `models/` へ自動で取得され、SHA-256を検証する。
- 既定は倍率x2とx4。時間に余裕があれば `--scales 2 4 8` でも実行する（ESPCN・EDSRのx8は、x4とx2の組み合わせになる）。
- EDSRは1回あたり数百ms〜数秒かかる。精度の上限の参考として、余裕があれば `--models espcn lapsrn edsr` で別の出力名で実行する。
- 標本が0件というエラーになったら、`--min-time`（遠い鍵が写り始める秒数）を記録に合わせて変える。

## 4. 結果をまとめてpushする

`docs/SR_MODEL_COMPARISON_202611.md` を作り、次を書く。

1. 使った記録・`predictions.csv`・検出器（パス）、実行したコマンド、PCのCPU・GPU
2. 検出器ごとの表（遠い実映像、1/4縮小、1/6縮小の各グループの再検出数と平均信頼度）
3. 方法ごとの拡大処理の時間（中央値、4スレッド）
4. 結論：現行のFSRCNN・bicubic・切り出しのみに比べて、ESPCN・LapSRNに**処理時間に見合う上積みがあるか**。2つの検出器で順位が一致するか（一致しなければ、検出器1つに依存した結果として扱う）
5. 注意：正解は比較実行のbaselineの検出で、人が確認した正解枠ではない。鍵の検出器はbicubicとFSRCNNの拡大画像を混ぜて学習しているので、ほかのモデルに不利

`outputs/` と `models/` はコミットしない（`.gitignore` 済み）。文書だけをコミットして、同じブランチへpushする。

```powershell
git add docs/SR_MODEL_COMPARISON_202611.md
git commit -m "Record comparison of super-resolution models on key recordings"
git push origin claude/current-situation-fl9j84
```
