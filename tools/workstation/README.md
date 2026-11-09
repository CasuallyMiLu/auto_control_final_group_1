# 偵測模型微調（任一台有 NVIDIA GPU 的 Linux 工作站，沿用 Lab2 的 ml/ 訓練環境；我們用 RTX 4090）

1. 6 支期末影片放 `~/ml/videos_final/`。
2. 本機用背景相減跑出每幀框：`python tools/test_vision_video.py --video <mp4> --kind circle|square|path --out <dir>`（CSV 含 bx0..by1），上傳到 `~/ml/final_bsm/`。
   （原本用 Lab2 的 SAM2 自動標註，但這批固定鏡頭影片 SAM2 第一個提示框到整塊板子，加手動提示後又卡在 1 幀 chunk 的無窮迴圈，所以改用穩定的背景相減框當標註。）
3. `~/ml/.venv/bin/python final_build_ds.py`：每 3 幀取 1 幀、前後 2 幀都有偵測且移動 < 30 px 才收；雷射影片全部當負樣本；加上 Lab2 原資料集。驗證集 = F_circle2 + F_path1（沒參與訓練）+ Lab2 val。去掉框大小離群（> 1.5 倍中位數）259 張。
4. `~/ml/.venv/bin/python final_train.py`：從 Lab2 的 y26s_det_960 best.pt 微調 60 epochs（early stop 57），flipud 0.5、degrees 10（攝影機可能轉向）。
5. 結果：mAP50 0.966、mAP50-95 0.853。`best.pt` → `final/models/final_det/`，本機 `tools/build_engine.py --runs final_det` 編 TensorRT。

本機對沒訓練過的影片：circle2 偵測 97.5%（Lab2 模型只有 15%）、與背景相減位置中位差 1 px；純雷射影片 0 個誤偵測。
