# auto_control_final_group_1

自動控制實驗 115-1 Final Project（第 1 組）：DJI Tello 以外接攝影機回授，**主題 1 循跡飛行**（圓形 / 方形）與
**主題 2 循跡導引計時賽**（紅線循跡 + 雷射筆導引 + 降落）。

**主程式是 MATLAB**（`matlab/`）。Python 版本（`theme1.py`…）功能相同，保留做對照，說明在 [docs/python_version.md](docs/python_version.md)。

---

## 快速開始（MATLAB）

```matlab
cd matlab
theme1('circle', 'sim', true)     % 先模擬：不用無人機、不用攝影機
theme1('circle')                  % 主題 1 圓形
theme1('square')                  % 主題 1 方形
theme2()                          % 主題 2
sysid_flight('circle')            % 系統辨識飛行（第一次飛先做這個）
```

飛之前：
1. 筆電 Wi-Fi 連 `TELLO-xxxxxx`；Windows 防火牆跳出來時允許 MATLAB（UDP 8889 / 8890）。
2. 插好外接攝影機（距牆約 450 cm、高約 130 cm，可放大、可轉 90° 裝）。程式會自己找「看得到板子」的那台攝影機、自己量比例和方向。
3. 無人機放起點，**起飛後要在畫面裡**：主題 1 在軌道右下方地上（離中心往右約 50–70 cm）；主題 2 在 A 點正下方。機頭朝牆。
4. 執行後先出現預覽視窗：綠線 = 參考路徑。確認無誤按 **G** 起飛。
5. 飛行中：**SPACE / q = 降落**、p = 暫停懸停、**ESC = 緊急停槳**、L = 立刻降落（主題 2 用）。
6. 每次飛完結果在 `runs/<日期時間>_<程式>_matlab/`：`log.csv`、`metrics.md`（時域規格）、`fig_*.png`。

---

## 第一次安裝

**MATLAB**（R2023b 以上，我們用 R2026a）：Image Processing、Computer Vision、Control System、System Identification Toolbox。
- 建議另外裝 Add-Ons → Get Hardware Support Packages → *MATLAB Support Package for USB Webcams*（沒裝也能跑，會改用 Python 讀攝影機）。
- Tello 用 UDP 文字指令直接控制（和助教 `ryzeRC.m` 的 `rc a b c d` 相同），**不需要** Ryze support package。

**Python**（YOLO 偵測、Python 攝影機、Python 版本才需要；Windows + NVIDIA GPU）：
```bash
python -m venv --system-site-packages .venv          # 系統要先有 CUDA 版 torch，沒有就在 venv 裡裝 torch (cu124/cu126)
.venv/Scripts/python -m pip install ultralytics==8.4.178 "tensorrt-cu12<11"
.venv/Scripts/python -m pip install --no-deps --ignore-installed "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python tools/build_engine.py --runs final_det     # 每台電腦編一次 TensorRT（約 10 分鐘）
```
（第三行只在 `torchvision::nms ... CUDA backend` 錯誤時需要：系統的 torchvision 是 CPU 版、和 CUDA torch 不合。版本要對應自己的 torch。）

MATLAB 會自動用 `.venv/Scripts/python.exe`（`pyenv` OutOfProcess）。

### 影像處理模式
| 選項 | 內容 |
|---|---|
| `'vision','yolo'`（用 Python 攝影機時的預設） | Python 端 YOLO26（Lab2 模型 + 這次影片微調）+ 背景相減修正 + Kalman，TensorRT 每幀約 15 ms |
| `'vision','bsm'`（用 MATLAB webcam 時的預設） | 純 MATLAB 背景相減（比背景暗的色塊，任一色板），在我們影片上偵測率 91–96 % |

---

## 流程：辨識 → 設計 → 飛行 → 微調

```matlab
cd matlab
sysid_flight('circle')                                   % 1. 約 100 秒：懸停 + X/Y 開迴路 doublet + PRBS
sysid_tello('../runs/<sysid 資料夾>', true)              % 2. procest P1DI + tfest（Lab1 做法）-> config/model.json
design_pid([], 60, true)                                 % 3. pidtune（Lab3 做法，PM 60°）-> config/gains.json
theme1('circle'); theme1('square'); theme2()             % 4. 飛
theme1('circle', 'set', {'path.v_max=30', 'cross.x.Kp=2.2'})   % 5. 微調（不用改程式）
```
Python 交叉驗證辨識：`.venv/Scripts/python tools/sysid_fit.py runs/<sysid 資料夾>`（同模型 + 每段漂移項）。

常用參數（`config/gains.json`）：`path.v_max` 巡航速度 cm/s、`path.a_lat` 轉彎、`cross.x/y.Kp/Ki/Kd/N` 循跡 PID、
`point.v_max` 雷射階段最高速、`predict_s` 延遲補償、`actuator.rc_max` rc 上限（預設 60 ≈ 0.6 m/s）。
場景設定（`config/rig.json`）：攝影機、方向、起點、繞行方向、主題 2 降落方式、安全範圍。

---

## 架構（對應講義控制系統架構圖）

```
攝影機 H ──► Image Processing ──► Transform ──► Tracking Algorithm ──► Controller C ──► rc a b c d ──► Tello P
          背景相減 / YOLO26       px -> 牆面 cm    最近點、切線、法線     PID (cross-track)    UDP 8889
          + Kalman               (每次自動校正)    + 速度規劃 (前饋)      + B⁻¹ 轉 rc
```

每一幀：影像 → 牆面座標 → 控制器 → rc 指令 → 安全檢查 → 紀錄 → 儀表板（**電腦同步動態顯示**：轉正的畫面 + 誤差與指令即時曲線）。

- **自動校正**：每次啟動從畫面找紅色軌道、白色中心線（150 cm）或紅線寬（20 cm），所以攝影機放大多少都不用設定（約 0.20 cm/px）。
- **自動轉正**：主題 2 由軌道形狀（B 端旁有藍色方塊）判斷上下；主題 1 先看地板在哪一側，起飛後往上爬 0.7 s 量實際「上」的方向、
  左右推一下確認 rc a 正負號（機頭朝攝影機也會自動修正）。
- **控制**：每軸 plant 視為 P(s) = e^{-Ls} / (s(τs+1))（rc → 牆面速度增益 Kv 由 B 矩陣反算），PID 由 pidtune 設計。
- **主題 1**：起飛 → 停在起點外 25 cm（固定大小的步階輸入）→ 放開開始循跡，一圈後降落。方形從下邊中點、圓形從右下出發，逆時針。
- **主題 2**：紅線 A→B 循跡 → 跟雷射點。**黑色障礙物完全不參與控制**（只畫在圖上），繞過障礙物是雷射操作者的工作。
  降落觸發 `config/rig.json` → `theme2.land_trigger`：`zone`（預設：一進入下方藍色方塊上空就降落）、`laser_off`（關掉雷射就降落）、`key`（按 L）。
- **安全**：看不到無人機 > 0.4 s 懸停、> 3 s 降落；飛出工作範圍降落；程式出錯或 Ctrl-C 自動降落。

## 時域規格定義（程式計算，`matlab/+ft/analyze_run.m`）
循跡誤差 e_n(t) = 無人機到白線的有號垂直距離（cm）。從起點外 25 cm 放開就是步階：
上升時間（90 %→10 % 的 e₀）、最大超越量（穿過白線到另一側的最大距離，cm 與 %）、峰值時間、
安定時間（之後一直在 ±5 cm 內；另列 ±3 cm）、穩態誤差（之後的平均 e_n，有號）、RMS、最大、在紅線上比例。
方形另列四個角的最大誤差；主題 2 另列雷射追蹤距離與總時間。
牆面 cm 與無人機實際 cm 差 L/(L−d) 倍（d = 無人機離牆距離，由視大小估計），metrics 有換算值。

## 目前狀態（給組員）
- 全部程式（MATLAB 與 Python）都在**模擬**中從起飛跑到降落；**還沒有真機飛行資料**。
- `config/model.json`、`config/gains.json` 目前是**先驗值**（rc 100 ≈ 1 m/s、τ 0.35 s、L 0.18 s，pidtune 設計）。
  第一步：真機跑 `sysid_flight('circle')`，用辨識結果重新設計增益。
- MATLAB 模擬結果（`runs/*_matlab_sim_*`）：

| | e₀ | 上升 | 超越 | 安定(±5 cm) | 穩態誤差 | RMS | 時間 |
|---|---|---|---|---|---|---|---|
| 圓 | 24.1 cm | 0.80 s | 12.5 % | 1.12 s | +1.18 cm | 2.2 cm | 一圈 22.0 s |
| 方 | 25.6 cm | 1.00 s | 5.7 % | 1.24 s | +0.02 cm | 1.4 cm | 一圈 41.0 s，角落 ≤ 4.4 cm |
| 主題 2 | | | | | +0.63 cm | 1.9 cm | 總 34.8 s |

## 檔案結構
```
matlab/                 主程式
  theme1.m theme2.m sysid_flight.m      每個主題一支程式，從起飛到降落
  sysid_tello.m design_pid.m            系統辨識 / PID 設計，寫回 config/
  +ft/                  calibrate（自動校正）, path_*, PID, PathFollower, PointTracker, Actuator, VisionBSM, LaserMATLAB,
                        TelloUDP, SimTello, CamSim/CamWebcam/CamPython, Dashboard, run_mission, analyze_run
theme1.py theme2.py sysid_flight.py     Python 版本（對照）
ftrack/                 Python 套件（scene, vision, control, mission, analysis, sysid, matlab_bridge = MATLAB 呼叫的 Python 端）
config/                 rig.json（硬體/場景）、model.json（plant）、gains.json（PID），*_prior.json 是先驗值備份
models/final_det/       偵測模型 best.pt（微調後），models/lab2_y26s_det_960/ 是 Lab2 原模型
assets/                 三塊板子的照片 + 無人機貼圖（模擬用）
data/video_tracks/      6 支場景影片的逐幀追蹤 CSV（影片本身在組內雲端）
runs/                   模擬結果範例（Python *_sim_baseline、MATLAB *_matlab_sim_*）
tools/                  check_camera / check_tello（不起飛）、build_engine、sysid_fit、test_vision_video、workstation/（重訓模型）
docs/python_version.md  Python 版本完整說明
```

## 重建 / 繼續
- **偵測模型重訓**：`tools/workstation/README.md`（6 支影片 → 背景相減自動標註 → 從 Lab2 模型微調 YOLO26s）。
- **場景校正檢查**：`.venv/Scripts/python tools/check_calib.py --kind circle --src <圖或影片>`。
- **影片離線追蹤**：`tools/test_vision_video.py`、`tools/video_track_error.py`（產生 `data/video_tracks/`）。
- **攝影機 / Tello 連線檢查**（不起飛）：`tools/check_camera.py`、`tools/check_tello.py`。
