# Python 版本（對照用）：Tello 循跡飛行 + 雷射導引

每個主題一支程式，從啟動到降落：

| 程式 | 內容 |
|---|---|
| `sysid_flight.py --scene circle` | **系統辨識飛行**（Lab1 方法用在真機）：懸停在板子中央 → X、Y 軸開迴路 doublet + PRBS → 降落 |
| `theme1.py --scene circle` / `--scene square` | **主題 1**：起飛 → 接近白色中心線（步階響應）→ 繞一圈 → 降落 |
| `theme2.py` | **主題 2**：起飛 → 紅線 A→B 循跡 → 跟雷射點繞過障礙物 → 停在下方藍色方塊上空 0.6 s → 自動降落 |

全部加 `--sim` 可以不接無人機、不接攝影機做模擬（用真實板子照片 + 無人機貼圖跑完整的影像處理 → 控制 → 紀錄流程）。

## 架構（對應講義控制系統架構圖）

```
攝影機 H ──► Image Processing ──► Transform ──► Tracking Algorithm ──► Controller C ──► rc a b c d ──► Tello P
 (USB)     YOLO26 + 背景相減      px → 牆面 cm     最近點/切線/法線       PID (cross-track)    UDP 8889
           + Kalman (Lab2)       (自動校正)       速度規劃 (feed-forward)  + B⁻¹ 轉成 rc
```

- `ftrack/scene.py`：每次啟動自動從畫面找紅色軌道、白色中心線、藍色方塊、黑色障礙物，算出 px ↔ 牆面 cm（X 右、Y 上）。
  **攝影機轉幾度、放大多少都不用設定**（見下方「尺寸與方向」）。
- `ftrack/vision.py`：無人機位置。預設 YOLO26（Lab2 模型再用這次 6 支影片微調），抓不到 GPU/模型時自動改用背景相減（`--detector bsm` 也可強制）。雷射點偵測也在這裡。
- `ftrack/control.py`：PID（Simulink Parallel 形式、微分濾波、anti-windup、integral zone）、路徑追蹤、點追蹤（雷射）。
- `ftrack/mission.py`：共用流程、安全機制、紀錄、儀表板（**電腦同步動態顯示控制狀態**：轉正的畫面 + 誤差/指令即時曲線）。
- `ftrack/tello.py`：Tello SDK（跟助教 `ryzeRC.m` 的 `rc` 指令完全相同的 UDP 文字協定，Python 直接送，延遲比 MATLAB 物件低）。
- `ftrack/analysis.py`：每次飛完自動算時域規格、畫圖。
- `matlab/sysid_tello.m`、`matlab/design_pid.m`：系統辨識（procest P1DI + tfest，Lab1 做法）與 PID 設計（pidtune，Lab3 做法），寫回 `config/`。

## 安裝（已經裝好在這台筆電）

`final/.venv` 已建好（沿用系統的 CUDA PyTorch，裝了 ultralytics 8.4.178、CUDA 版 torchvision、TensorRT）。換電腦時：

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install ultralytics==8.4.178 "tensorrt-cu12<11"
.venv/Scripts/python tools/build_engine.py        # 每台電腦編一次 TensorRT（約 5–8 分鐘）
```

以下指令都在 repo 根目錄執行，用 `.venv/Scripts/python`。

## 飛行日流程

### 0. 飛行前檢查
1. 攝影機架好（跟拍影片時一樣：距牆約 450 cm、高 130 cm、可以放大），畫面要看得到整個軌道。轉 90° 裝沒關係。
   攝影機編號預設 `auto`：程式會自己找「看得到這塊板子」的那台（筆電內建鏡頭會被跳過）。想先確認：`.venv/Scripts/python tools/check_camera.py`。
2. 筆電 Wi-Fi 連 `TELLO-xxxxxx`。第一次執行若 Windows 防火牆跳出來，允許 Python（UDP 8889/8890）。**MATLAB 的 ryze 物件不要同時開著**（會佔用同一個埠）。
3. 電池 > 30%（程式會檢查，不夠會拒絕起飛）。先跑 `.venv/Scripts/python tools/check_tello.py`（不會起飛）確認連線、電量、狀態回傳。
4. 無人機放在起點，**起飛後一定要在畫面裡**（放大 + 轉 90° 後左右只看得到中心 ±105 cm 左右）：
   - 主題 1：圓/方形**右下方**、離中心往右約 50–70 cm 的地上（紅線正下方附近，不要放到軌道外太遠）。
   - 主題 2：**A 點正下方**地上。
   - **機頭朝牆**（朝向攝影機也可以，起飛後的方向測試會自動修正）。
5. 確認畫面：程式啟動後先出現預覽視窗，綠線 = 參考路徑，黃色箭頭 = 起點與飛行方向，粉紅/橘色箭頭 = 上/右。都對就按 **G** 起飛，不對按 **Q** 離開。

### 1. 系統辨識飛行（先做這個，參數就是從這裡來）
```bash
.venv/Scripts/python sysid_flight.py --scene circle
```
約 100 秒飛行。結束後把 `runs/<日期時間>_sysid_circle/` 整個資料夾傳給我（或自己跑下一步）。建議做 2 次（換電池前後），辨識結果互相驗證。

辨識與設計（我會做；你要自己跑也可以）：
```bash
.venv/Scripts/python tools/sysid_fit.py runs/<sysid run> --write      # Python 辨識 → config/model.json
matlab -batch "cd matlab; sysid_tello('../runs/<sysid run>', true); design_pid([], 60, true)"   # MATLAB 辨識 + pidtune → config/gains.json
```

### 2. 主題 1 / 主題 2
起飛後流程：找到無人機 → 往上 0.7 s、左右推一下（確認方向，約 3 s）→（主題 1）飛到起點外 25 cm 停住 → 放開開始循跡，這一步就是固定大小 25 cm 的**步階輸入**，每次的上升時間、超越量、安定時間可以互相比較 → 一圈後降落。
方形從**下邊中點**出發（第一個角之前有乾淨的步階響應），圓形從右下（-45°）出發，都是逆時針（`config/rig.json` 的 `closed_path` 可改）。
```bash
.venv/Scripts/python theme1.py --scene circle
.venv/Scripts/python theme1.py --scene square
.venv/Scripts/python theme2.py
```
主題 2 的雷射：無人機到 B 點後，雷射筆點在無人機前方約 20–30 cm 的牆上，帶它從障礙物左邊繞下去，最後點在下方藍色方塊上，無人機在方塊上停 0.6 s 就自動降落（降到平台上）。

### 3. 微調
不用改程式，加參數覆寫：
```bash
.venv/Scripts/python theme1.py --scene circle --set path.v_max=30 --set cross.x.Kp=2.2 --tag fast
```
常用：`path.v_max`（巡航速度 cm/s）、`path.a_lat`（轉彎）、`cross.x.Kp / Ki / Kd`、`point.v_max`（雷射階段最高速）、`predict_s`（延遲補償秒數）。

## 安全
- 儀表板視窗：**SPACE 或 q = 立刻降落**，p = 暫停（原地懸停）/繼續，ESC = 緊急停槳（會掉下來，只在危險時用）。
- 看不到無人機 > 0.4 s → 原地懸停；> 3 s → 自動降落。
- 飛出工作範圍（軌道外 45 cm）→ 自動降落。程式出錯或 Ctrl-C → 自動降落。
- rc 指令最大 ±60（約 0.6 m/s）。

## 每次飛行的輸出 `runs/<日期時間>_<程式>/`
| 檔案 | 內容 |
|---|---|
| `log.csv` | 每一幀：時間、階段、像素與 cm 位置、誤差、參考點、rc 指令、雷射、Tello 狀態（電量、高度、速度、姿態）、延遲 |
| `meta.json` | 當次的場景校正、參數、模型、事件時間表 |
| `metrics.md` / `metrics.json` | 時域規格（下方定義）|
| `fig_trajectory.png`, `fig_error_time.png`, `fig_overview.png`, `fig_theme2.png` | 實際 vs 理想路徑、誤差對時間、指令 |
| `dashboard.mp4`, `camera_raw.mp4` | 儀表板錄影、原始攝影機錄影（報告影片用）|

## 尺寸與方向（攝影機放大、轉 90° 的修正）
- **放大**：不用知道放大倍率。程式每次啟動量白色中心線（直徑/邊長 150 cm）或紅線寬度（20 cm），直接得到 cm/px（目前約 0.20 cm/px）。拍攝那天量到紅帶寬 19.1–19.5 cm，和規格 20 cm 一致。
- **轉 90°**：主題 2 由軌道形狀判斷（B 端旁邊有藍色方塊 → 那邊是上）；主題 1 的圓/方形是對稱的，先看地板在畫面哪一邊，起飛後再往上爬 0.7 s 量實際「上」的方向確認（錯了會自動轉正），接著左右推一下確認 rc a 的正負號（機頭朝攝影機也能自動修正）。
- **牆面 cm vs 無人機實際 cm**：無人機飛在牆前方 d cm，攝影機距牆 L = 450 cm，影像上量到的距離（換算到牆面）比無人機實際移動大 L/(L−d) 倍。控制和規格以牆面座標計算（題目定義「以影像中心點代表四旋翼」並對準牆上軌道）；`metrics.md` 會用無人機的視大小估 d（例：d ≈ 30 cm → 實際 = 牆面 × 0.93）並另外列出換算到無人機平面的誤差。

## 時域規格定義（`ftrack/analysis.py`，全部程式計算）
循跡誤差 e_n(t)：無人機到參考路徑（白線）的有號垂直距離（cm，左正）。從起飛點接近軌道就是一個大小 e₀ 的步階：
- 上升時間 t_r：|e_n| 從 90% e₀ 降到 10% e₀ 的時間
- 最大超越量 M_p：穿過白線後到另一側的最大距離（cm 與 % of e₀），峰值時間 t_p
- 安定時間 t_s：之後一直保持在 ±5 cm（紅線寬的 1/4）內的起始時間；另列 ±3 cm
- 穩態誤差 e_ss：t > t_s 的平均 e_n（有號，代表偏向哪一側）；另列 RMS、最大、P95、在紅線上比例（|e_n| ≤ 10 cm）
- 方形另列四個角的最大誤差；主題 2 另列雷射追蹤距離與總時間

## 模擬目前的結果（先驗模型 Kv=1 cm/s/rc、τ=0.35 s、L=0.18 s，pidtune 增益，YOLO 在迴路裡）
`runs/*_sim_baseline/`（每個資料夾有圖、metrics.md、dashboard.mp4）：

| | e₀ | 上升 | 超越 | 安定(±5 cm) | 穩態誤差 | 穩態 RMS | 時間 |
|---|---|---|---|---|---|---|---|
| 圓 | 23.6 cm | 0.84 s | 11.5 % | 1.04 s | −0.60 cm | 1.8 cm | 一圈 21.9 s |
| 方 | 25.6 cm | 0.92 s | 6.6 % | 1.16 s | +0.45 cm | 2.0 cm | 一圈 34.4 s，四角最大 4.4 cm |
| 主題 2 | | | | | +0.98 cm | 1.8 cm | 循跡 17.6 s，總 30.5 s |

系統辨識流程的驗證：模擬裡故意放一個不同的「真實」模型（Kv 0.85/1.15、τ 0.45/0.25、L 0.16，加隨機陣風），
`sysid_flight.py` + `tools/sysid_fit.py` 辨識出 Kv 0.81/1.21、τ 0.40/0.25，驗證集 fit 85–95 %（無陣風時誤差 < 1 %）。
MATLAB `sysid_tello.m` 在無陣風資料上得到 Kv 0.856/1.052、τ 0.42/0.28，驗證 fit 88–96 %。

這些是**模擬**，真機要等系統辨識後重新設計增益再飛。

## 偵測模型
`models/final_det/`：Lab2 的 YOLO26s 用這次 6 支影片微調（做法見 `tools/workstation/README.md`）。沒參與訓練的圓形影片偵測率 97.5 %（原 Lab2 模型 15 %），純雷射影片 0 誤判，TensorRT 每幀約 15–25 ms。
