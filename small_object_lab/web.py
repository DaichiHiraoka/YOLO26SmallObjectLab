from __future__ import annotations


COMMON_STYLE = """
:root {
  color-scheme: dark;
  --bg: #07111f;
  --panel: #101d2d;
  --line: #26384d;
  --text: #e8f0f7;
  --muted: #93a4b8;
  --accent: #36d399;
  --warn: #fbbf24;
  --danger: #fb7185;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: radial-gradient(circle at top, #12243b, var(--bg) 48%);
  color: var(--text); font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { width: min(1500px, 96vw); margin: 0 auto; padding: 24px 0 40px; }
h1 { margin: 0; font-size: clamp(1.45rem, 3vw, 2.35rem); letter-spacing: -.03em; }
h2 { margin: 0 0 14px; font-size: 1.05rem; }
p { color: var(--muted); }
.eyebrow { color: var(--accent); font-weight: 750; font-size: .8rem; letter-spacing: .14em; }
.top { display: flex; justify-content: space-between; align-items: start; gap: 20px; margin-bottom: 18px; }
.badge { padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); }
.badge.live { border-color: var(--accent); color: var(--accent); }
.grid { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(310px, .75fr); gap: 16px; align-items: start; }
.panel { background: color-mix(in srgb, var(--panel) 92%, transparent); border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 18px 60px #0005; }
.video { width: 100%; min-height: 320px; display: block; object-fit: contain; border-radius: 10px; background: #02060b; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }
.metric { padding: 11px; background: #091421; border: 1px solid var(--line); border-radius: 10px; }
.metric span { display: block; color: var(--muted); font-size: .72rem; margin-bottom: 5px; }
.metric strong { font-size: 1.03rem; }
.record { border-color: #316c59; background: #0b2922; }
.stack { display: grid; gap: 16px; }
label { display: block; margin: 10px 0 5px; color: var(--muted); font-size: .8rem; }
input, select, button, a.button {
  width: 100%; border-radius: 9px; border: 1px solid var(--line); background: #08131f;
  color: var(--text); padding: 10px 11px; font: inherit; text-decoration: none;
}
button, a.button { cursor: pointer; text-align: center; background: #163047; font-weight: 700; }
button.primary { background: var(--accent); color: #052017; border-color: var(--accent); }
button:hover, a.button:hover { filter: brightness(1.15); }
button:disabled { opacity: .45; cursor: default; filter: none; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.check { display: flex; align-items: center; gap: 9px; }
.check input { width: auto; }
.actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
.small { font-size: .78rem; color: var(--muted); overflow-wrap: anywhere; }
.error { color: var(--danger); min-height: 1.2em; }
.table-wrap { max-height: 200px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: .8rem; }
th, td { padding: 7px 5px; border-bottom: 1px solid var(--line); text-align: right; }
th:first-child, td:first-child { text-align: left; }
code { color: #b7f7df; }
@media (max-width: 900px) {
  .top { flex-direction: column; }
  .grid { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: 1fr 1fr; }
}
"""


# Shared by the PC dashboard and the phone page. Pages define poll() and #error.
RECORDING_SCRIPT = """
let recordingBusy = false;
const RECORDING_LABELS = {waiting: "記録中（映像待ち）", recording: "記録中", limit_reached: "容量上限で停止", error: "記録エラー"};
function showError(text) { const element = document.querySelector("#error"); if (element) element.textContent = text; }
async function post(path, body = {}) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "request failed");
  return data;
}
function recordingSummary(r) {
  const last = r.last;
  if (!last) return {text: "記録していません", warn: false, busy: recordingBusy, state: "記録していません"};
  let state = RECORDING_LABELS[last.state] || last.state;
  if (last.closing) state = "記録を確定中";
  else if (!r.active) state = last.complete ? "記録完了" : "記録エラー（未完了）";
  if (last.state === "limit_reached" && !last.closing) state += "（容量上限）";
  const error = last.errors.length ? ` / ${last.errors[last.errors.length - 1].message}` : "";
  const text = `${state} / ${last.session_id} / ${last.duration_s.toFixed(1)}秒 / 保存 ${last.saved_frames}枚 / 欠落 ${last.dropped_frames}枚 / ${(last.bytes_written / 1024 ** 3).toFixed(2)} GiB${error}`;
  const warn = last.dropped_frames > 0 || last.errors.length > 0 || last.state === "limit_reached";
  return {text, warn, busy: recordingBusy || last.closing, state};
}
async function recordingAction(path) {
  recordingBusy = true;
  try { await post(path); showError(""); }
  catch (e) { showError(e.message); }
  finally { recordingBusy = false; poll(); }
}
"""


DASHBOARD_HTML = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO26 Small Object Lab</title><style>{COMMON_STYLE}</style></head>
<body><main>
  <div class="top">
    <div><div class="eyebrow">YOLO26 / TAILSCALE CAMERA LAB</div>
      <h1>どこまで小さく見つけられるか。</h1>
      <p>黄色は未確定、緑は連続検出済み。記録値は入力映像上のbboxサイズです。</p></div>
    <div id="connection" class="badge">android-camera-streamer待機中</div>
  </div>
  <div class="grid">
    <section class="panel">
      <img id="stream" class="video" alt="推論映像">
      <div class="metrics">
        <div class="metric"><span>入力解像度</span><strong id="resolution">—</strong></div>
        <div class="metric"><span>推論</span><strong id="speed">—</strong></div>
        <div class="metric"><span>現在の最小bbox</span><strong id="current">—</strong></div>
        <div class="metric record"><span>セッション最小（確定）</span><strong id="record">—</strong></div>
      </div>
      <div class="actions">
        <button onclick="action('/api/reset')">最小記録をリセット</button>
        <button onclick="action('/api/snapshot')">画像を保存</button>
        <a class="button" href="/api/detections.csv">検出CSV</a>
        <a class="button" href="/api/frames.csv">フレームCSV</a>
      </div>
      <p id="autosave" class="small">15秒ごとに画像と測定状態を自動保存します。</p>
    </section>
    <aside class="stack">
      <section class="panel">
        <h2>元映像の記録</h2>
        <div class="row">
          <button id="recStart" class="primary" onclick="recordingAction('/api/recording/start')">記録開始</button>
          <button id="recStop" onclick="recordingAction('/api/recording/stop')" disabled>記録終了</button>
        </div>
        <p id="recordingState" class="small">記録していません。開始から終了までが1つの撮影セッションです（1回につき最大10 GiB）。</p>
        <p id="remoteUrl" class="small"></p>
      </section>
      <section class="panel">
        <h2>推論設定</h2>
        <label>YOLO入力サイズ（大きいほど小物体に有利・低速）</label>
        <select id="imgsz">
          <option>320</option><option>480</option><option>640</option>
          <option selected>960</option><option>1280</option><option>1536</option>
        </select>
        <div class="row">
          <div><label>confidence</label><input id="confidence" type="number" min=".01" max=".99" step=".01" value=".20"></div>
          <div><label>確定までの連続フレーム</label><input id="stable" type="number" min="1" max="30" value="3"></div>
        </div>
        <label>対象クラス（英語、カンマ区切り。空欄は全クラス）</label>
        <input id="classes" value="cup" placeholder="cell phone, bottle, cup">
        <label class="check"><input id="srEnabled" type="checkbox" checked>
          不安定兆候時に最大bboxを局所超解像</label>
        <div class="row">
          <div><label>超解像を始めるconfidence以下</label>
            <input id="srConfidence" type="number" min=".01" max=".99" step=".01" value=".45"></div>
          <div><label>超解像を始めるbbox面積%以下</label>
            <input id="srArea" type="number" min=".01" max="100" step=".01" value="1.00"></div>
        </div>
        <div class="row">
          <div><label>超解像後の目標短辺px</label>
            <input id="srTargetMinPx" type="number" min="16" max="320" step="8" value="96"></div>
          <div><label>超解像の最大倍率</label>
            <select id="srMaxScale"><option>2</option><option>4</option><option selected>8</option></select></div>
        </div>
        <button class="primary" style="margin-top:12px" onclick="saveSettings()">設定を反映</button>
        <p id="srState" class="small"></p>
        <div id="error" class="error small"></div>
      </section>
      <section class="panel">
        <h2>実物条件を記録</h2>
        <label>物体名</label><input id="objectName" placeholder="USBメモリ">
        <div class="row">
          <div><label>実幅 mm（任意）</label><input id="physicalWidth" type="number" min="0" step=".1"></div>
          <div><label>距離 cm（任意）</label><input id="distance" type="number" min="0" step=".1"></div>
        </div>
        <button style="margin-top:12px" onclick="recordMeasurement()">現在値を測定CSVへ記録</button>
        <a class="button" style="display:block;margin-top:8px" href="/api/measurements.csv">測定CSV</a>
      </section>
      <section class="panel">
        <h2>現在の検出</h2>
        <div class="table-wrap"><table><thead><tr><th>class</th><th>conf</th><th>bbox</th><th>面積%</th><th>連続</th></tr></thead>
        <tbody id="detections"><tr><td colspan="5">検出待ち</td></tr></tbody></table></div>
        <p id="model" class="small"></p><p id="output" class="small"></p>
      </section>
    </aside>
  </div>
</main>
<script>
let initialized = false;
function boxText(d) {{ return d ? `${{d.width_px}}×${{d.height_px}} px (${{d.area_pct}}%)` : "—"; }}
async function poll() {{
  try {{
    const s = await (await fetch("/api/status", {{cache:"no-store"}})).json();
    const badge = document.querySelector("#connection");
    badge.textContent = s.camera_connected ? "スマホ接続中" : s.status;
    badge.className = s.camera_connected ? "badge live" : "badge";
    document.querySelector("#resolution").textContent = s.source_width ? `${{s.source_width}}×${{s.source_height}}` : "—";
    document.querySelector("#speed").textContent = s.processed_frames ? `${{s.inference_ms}} ms / ${{s.inference_fps}} FPS` : "—";
    document.querySelector("#current").textContent = boxText(s.smallest_current);
    document.querySelector("#record").textContent = boxText(s.smallest_confirmed);
    document.querySelector("#model").textContent = `${{s.model}} / ${{s.device}} / ${{s.source_mode}} / ${{s.processed_frames}} frames`;
    document.querySelector("#output").textContent = `保存先: ${{s.output_dir}}`;
    document.querySelector("#autosave").textContent =
      `${{s.auto_save_interval_sec}}秒ごとに自動保存 / 次回 ${{s.auto_save_in_sec}}秒後 / 保存 ${{s.auto_save_count}}件`;
    const recording = recordingSummary(s.recording);
    const recordingElement = document.querySelector("#recordingState");
    recordingElement.textContent = recording.text;
    recordingElement.className = recording.warn ? "small error" : "small";
    document.querySelector("#recStart").disabled = recording.busy || s.recording.active;
    document.querySelector("#recStop").disabled = recording.busy || !s.recording.active;
    document.querySelector("#remoteUrl").textContent = s.remote_url
      ? `スマホから操作: ${{s.remote_url}}`
      : "スマホから操作: 無効（PCのTailscale IPで待ち受けていません）";
    const sr = s.super_resolution;
    document.querySelector("#srState").textContent = sr.active
      ? `${{sr.model}} x${{sr.scale}}実行 (${{sr.reason}}) / 試行 ${{sr.scales_tried.join("→")}} / ${{sr.inference_ms}} ms / 採用 ${{sr.adoptions}}/${{sr.attempts}}`
      : `局所超解像待機 / x2〜x${{s.settings.sr_max_scale}} / 対象: 最大bbox / 採用 ${{sr.adoptions}}/${{sr.attempts}}`;
    document.querySelector("#error").textContent = s.error || sr.error || "";
    document.querySelector("#detections").innerHTML = s.detections.length ? s.detections.map(d =>
      `<tr><td>${{d.label}}${{d.source === "super_resolution" ? " (SR)" : ""}}</td><td>${{d.confidence}}</td><td>${{d.width_px}}×${{d.height_px}}</td><td>${{d.area_pct}}</td><td>${{d.streak}}/${{s.settings.stable_frames}}</td></tr>`
    ).join("") : '<tr><td colspan="5">検出なし</td></tr>';
    if (!initialized) {{
      document.querySelector("#imgsz").value = s.settings.imgsz;
      document.querySelector("#confidence").value = s.settings.confidence;
      document.querySelector("#stable").value = s.settings.stable_frames;
      document.querySelector("#classes").value = s.settings.target_classes;
      document.querySelector("#srEnabled").checked = s.settings.super_resolution_enabled;
      document.querySelector("#srConfidence").value = s.settings.sr_trigger_confidence;
      document.querySelector("#srArea").value = s.settings.sr_trigger_area_pct;
      document.querySelector("#srTargetMinPx").value = s.settings.sr_target_min_px;
      document.querySelector("#srMaxScale").value = s.settings.sr_max_scale;
      initialized = true;
    }}
  }} catch (e) {{ document.querySelector("#error").textContent = "状態取得失敗: " + e; }}
}}
{RECORDING_SCRIPT}
async function saveSettings() {{
  try {{
    await post("/api/settings", {{
      imgsz:+document.querySelector("#imgsz").value,
      confidence:+document.querySelector("#confidence").value,
      stable_frames:+document.querySelector("#stable").value,
      target_classes:document.querySelector("#classes").value,
      super_resolution_enabled:document.querySelector("#srEnabled").checked,
      sr_trigger_confidence:+document.querySelector("#srConfidence").value,
      sr_trigger_area_pct:+document.querySelector("#srArea").value,
      sr_target_min_px:+document.querySelector("#srTargetMinPx").value,
      sr_max_scale:+document.querySelector("#srMaxScale").value
    }}); document.querySelector("#error").textContent = "";
  }} catch(e) {{ document.querySelector("#error").textContent = e; }}
}}
async function action(path) {{
  try {{ await post(path); }} catch(e) {{ document.querySelector("#error").textContent = e; }}
}}
async function recordMeasurement() {{
  try {{
    await post("/api/record", {{
      object_name:document.querySelector("#objectName").value,
      physical_width_mm:document.querySelector("#physicalWidth").value,
      camera_distance_cm:document.querySelector("#distance").value
    }});
    document.querySelector("#error").textContent = "測定値を記録しました";
  }} catch(e) {{ document.querySelector("#error").textContent = e; }}
}}
// On the PC itself show every frame; from the phone (Tailscale) cap it to save mobile data.
const onThisPc = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
document.querySelector("#stream").src = onThisPc ? "/api/stream" : "/api/stream?fps=10";
setInterval(poll, 500); poll();
</script></body></html>"""


# Phone page, opened over Tailscale: http://<PC Tailscale IP>:8765/remote
REMOTE_HTML = (
    """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>記録操作 - YOLO26 Small Object Lab</title><style>"""
    + COMMON_STYLE
    + """
main { width: min(560px, 100%); padding: 16px; }
.state { font-size: 1.6rem; font-weight: 800; margin: 4px 0; }
.state.live { color: var(--danger); }
.big { font-size: 1.35rem; padding: 22px 12px; margin-top: 12px; }
#result { display: block; width: 100%; margin-top: 12px; border-radius: 8px; background: #000; }
#result[hidden] { display: none; }
</style></head>
<body><main>
  <div class="eyebrow">YOLO26 SMALL OBJECT LAB</div>
  <h1>元映像の記録</h1>
  <section class="panel" style="margin-top:14px">
    <div id="state" class="state">接続中…</div>
    <p id="detail" class="small"></p>
    <p id="camera" class="small"></p>
    <button id="recStart" class="primary big" onclick="recordingAction('/api/recording/start')" disabled>記録開始</button>
    <button id="recStop" class="big" onclick="recordingAction('/api/recording/stop')" disabled>記録終了</button>
    <div id="error" class="error small"></div>
  </section>
  <section class="panel" style="margin-top:14px">
    <button id="resultToggle" onclick="toggleResult()">検出結果を表示</button>
    <p class="small">PCが枠を描いた映像です（最大10fps）。モバイル回線では通信量に注意してください。</p>
    <img id="result" alt="PCの検出結果" hidden>
  </section>
</main>
<script>"""
    + RECORDING_SCRIPT
    + """
async function poll() {
  try {
    const s = await (await fetch("/api/status", {cache: "no-store"})).json();
    const recording = recordingSummary(s.recording);
    const state = document.querySelector("#state");
    state.textContent = s.recording.active ? "● " + recording.state : recording.state;
    state.className = s.recording.active ? "state live" : "state";
    const detail = document.querySelector("#detail");
    detail.textContent = s.recording.last ? recording.text : "開始から終了までが1つの撮影セッションです。";
    detail.className = recording.warn ? "small error" : "small";
    document.querySelector("#camera").textContent = s.camera_connected ? "カメラ映像: 受信中" : `カメラ映像: ${s.status}`;
    document.querySelector("#recStart").disabled = recording.busy || s.recording.active;
    document.querySelector("#recStop").disabled = recording.busy || !s.recording.active;
  } catch (e) { showError("状態取得失敗: " + e); }
}
function toggleResult() {
  const image = document.querySelector("#result");
  const show = image.hidden;
  // Clearing src closes the stream, so no data flows while it is hidden.
  image.src = show ? "/api/stream?fps=10" : "";
  image.hidden = !show;
  document.querySelector("#resultToggle").textContent = show ? "検出結果を隠す" : "検出結果を表示";
}
setInterval(poll, 1000); poll();
</script></body></html>"""
)
