# Tello 真機系統辨識：離線操作單

適用於本專案的圓形板場景、MATLAB webcam + BSM 路線。使用外接 USB 攝影機，電腦 Wi-Fi 連 Tello。整個流程在本機執行，連上 Tello 後不需要網際網路。本操作單的指令供操作者逐段執行，沒有替你啟動真機。

> 2026-11-10 補充：實驗經驗、已知限制與後續改進見 [真機辨識交接](sysid_handoff_20261110.md)。目前只完成初步模型估算，尚未接受為正式控制模型。

## 0. 連上 Tello 前準備

- 保留這份文件在電腦。MATLAB、四個必要工具箱及 USB Webcams 支援套件應先準備好；目前這台電腦先前已能使用 webcamlist。
- 關閉會使用攝影機的視訊軟體；關閉其他 Tello 控制程式，避免搶用相同設備或 UDP 8889/8890。
- 相機架穩、拍到整張板及飛行範圍，起飛後仍需看到無人機。使用與 rig.json 一致的板尺寸（目前中心線尺度 150 cm、帶寬 20 cm）。
- 無人機依 README 放在圓板右下方的地面，機頭朝牆。預留移動空間，移開人員與障礙物，由熟悉設備的人協助監看。
- 使用電量充足的電池；程式目前的起飛最低門檻是 30%，這只是軟體門檻，沒有飛行中的低電量中止判斷。實測曾從 33% 降到 9%，不應把起飛檢查通過當作可完成整輪的保證。下輪採集前需先補上持續電量處理，並考慮分軸縮短單次實驗。
- 電腦 Wi-Fi 連 TELLO-xxxxxx。顯示「無網際網路」是此連線方式的預期情況；維持該 Wi-Fi。

## 1. 設定 MATLAB 專案位置

在 MATLAB Command Window 執行：

```matlab
projectRoot = '/Users/eason/Desktop/大四教科書/大四上/03-電工實驗（自動控制）/auto_control_final_group_1-main';
cd(fullfile(projectRoot, 'matlab'));
addpath(pwd);
disp(which('sysid_flight'));
disp(which('ft.run_mission'));
```

兩個路徑都應位於上述主要專案，不能指向之前的 /private/tmp 副本。不需要執行 restoredefaultpath。

## 2. 檢查 USB 攝影機（不會起飛）

```matlab
cameraNames = webcamlist;
disp(cameraNames);
```

依清單選外接鏡頭。下例的 1 請改成實際編號；這裡 MATLAB 編號從 1 開始：

```matlab
cameraNumber = 1;
cam = webcam(cameraNumber);
try
    figure;
    imshow(snapshot(cam));
catch cameraError
    clear cam;
    rethrow(cameraError);
end
clear cam;
```

確認鏡頭正確、板與飛行範圍可見，然後一定釋放 cam。若 Mac 要求相機權限，允許 MATLAB；若畫面不是外接鏡頭，換 cameraNumber 再測。

正式程式預設會自動找看得到這塊板的鏡頭。若正式預覽仍挑錯鏡頭，按 q 退出，先修正設定，不要起飛。可在主要專案 config/rig.json 把 camera.index 從 auto 改成外接鏡頭的零起算編號（即 cameraNumber-1）；其餘設定保持原值。注意這與 webcam(cameraNumber) 的一開始編號不同。

## 3. 檢查 Tello 通訊與電量（不會起飛）

以下只送 command 與查詢，不含 takeoff。必須關閉這個檢查物件，釋放 UDP 通訊埠後才能開正式程式。

```matlab
checkDrone = ft.TelloUDP('192.168.10.1');
try
    connected = checkDrone.connect();
    assert(connected, 'Tello 無回應：檢查 Wi-Fi、其他控制程式與通訊埠。');
    batteryPercent = checkDrone.battery();
    fprintf('Tello 電量：%.0f%%\n', batteryPercent);
catch connectionError
    checkDrone.close();
    clear checkDrone;
    rethrow(connectionError);
end
checkDrone.close();
clear checkDrone;
assert(isfinite(batteryPercent) && batteryPercent >= 30, ...
    '電量查詢失敗或低於 30%%，先處理後再繼續。');
```

若這一段失敗，先排除問題，不要直接跳到起飛指令。若 constructor 就報通訊埠占用，不會產生檢查物件；關閉其他程式或重啟 MATLAB 釋放舊連線後再試。

## 4. 啟動真機辨識，先停在預覽

這一段是【真機模式】。按 G 後會控制真機起飛，不能當成模擬執行。請逐段操作，不要把整份文件當成一個腳本執行。

```matlab
sysid_flight('circle', ...
    'sim', false, ...
    'camera', 'webcam', ...
    'vision', 'bsm', ...
    'yes', false, ...
    'show', true, ...
    'tag', 'real01');
```

預覽時確認鏡頭、綠色軌道、畫面方向及場地範圍正確。開始前不要移動攝影機。設定不對先按 q 結束。

相機可能旋轉安裝，程式先依場景猜方向，起飛後會短暫上升及左右測試再修正。這些動作也會使無人機移動，需保留空間並監看。軟體目前只有模擬驗證，不能把方向修正或保護機制視為真機安全保證。

## 5. 按 G 起飛、監看測試

點一下辨識儀表板，確保鍵盤焦點在該視窗，再按 G。

流程為 TAKEOFF → ACQUIRE → PROBE_UP / PROBE_SIDE → GOTO / EXCITE 交替 → LAND / DONE。程式會依序測 X 與 Y，送正反脈衝與隨機激振，在段落間回到辨識點。

- 人員全程監看；測試中不要同時用其他介面操控無人機，不要移動相機或遮擋視線。
- 預期約一至數分鐘，實際取決於追蹤及回中心的時間；以結束原因判斷完成，不以秒數判斷。
- 如果方向錯、追蹤位置明顯不合理、移向障礙物或離開視野，立即中止。資料可重收。
- 專案含追蹤遺失、越界與回中心超時處理，但仍需現場監看。

| 按鍵（焦點在儀表板） | 動作 |
|---|---|
| Space 或 q | 要求降落，一般中止使用 |
| p | 暫停／繼續；暫停不保證鎖定位置，且會影響測試時間，不宜把該次資料視為未中斷實驗 |
| Esc | 緊急停槳，立即失去升力；僅緊急時使用 |

若程式不回應，可在 MATLAB Command Window 用 Ctrl+C 中止；程式有嘗試降落與釋放設備的 cleanup，但通訊故障或強制結束時不保證有效，需目視確認並依現場接管程序處理。

正常應出現 LAND: identification program finished，以及 run saved in ...。先目視確認無人機已安全落地，再進行下列分析。

## 6. 找到這次真機紀錄

以下只挑 tag=real01 的最新 sysid 紀錄，排除之前的模擬結果：

```matlab
projectRoot = '/Users/eason/Desktop/大四教科書/大四上/03-電工實驗（自動控制）/auto_control_final_group_1-main';
cd(fullfile(projectRoot, 'matlab'));
runList = dir(fullfile(projectRoot, 'runs', '*_sysid_*_matlab_real01'));
assert(~isempty(runList), '找不到 real01 真機紀錄，請查看起飛程式顯示的儲存路徑。');
[~, latestIndex] = max([runList.datenum]);
runDir = fullfile(runList(latestIndex).folder, runList(latestIndex).name);
disp(runDir);
meta = jsondecode(fileread(fullfile(runDir, 'meta.json')));
assert(~meta.sim, '這是模擬資料，請重新選擇真機 run。');
fprintf('結束原因：%s\n', meta.land_reason);
```

如果你改了起飛指令的 tag，這裡的 real01 也要跟著改。若最新一次被取消或中途降落，程式仍可能建立資料夾；不要因為有資料夾就認定資料完整。

## 7. 檢查資料是否完成

```matlab
assert(strcmp(meta.land_reason, 'identification program finished'), ...
    '這次未完整結束；保留資料，但先不要當作完整的兩軸辨識。');
assert(isfile(fullfile(runDir, 'log.csv')), '缺少 log.csv。');
importOptions = detectImportOptions(fullfile(runDir, 'log.csv'), 'TextType', 'string');
importOptions = setvartype(importOptions, {'phase', 'excite_axis'}, 'string');
flightData = readtable(fullfile(runDir, 'log.csv'), importOptions);
fprintf('總紀錄：%d 列\n', height(flightData));
for axisName = ["x", "y"]
    selected = flightData.phase == "EXCITE" & flightData.excite_axis == axisName;
    fprintf('%s 激振紀錄：%d 列，追蹤有效 %d 列\n', ...
        upper(axisName), nnz(selected), nnz(selected & flightData.valid == 1));
    assert(nnz(selected) > 0, '缺少某一軸的激振資料。');
end
```

有效列數不是品質通過門檻；大量掉追蹤、錯誤定位或訊號截斷仍要另外檢查。若命令視窗顯示 block ... stopped，表示該段激振觸發範圍保護而截斷，即使最後 reason 顯示 finished 也要記錄這項限制，不要為求完整而任意放寬飛行範圍。

## 8. 離線辨識與檢查

這些指令不會飛行，也不需保持 Tello Wi-Fi。無人機落地且正式程式結束後，即可關閉 Tello、切回一般網路再執行，或繼續離線執行。

```matlab
M = sysid_tello(runDir, false);
assert(isfield(M, 'x') && isfield(M, 'y'), '有軸未成功辨識。');
assert(M.x.nValid > 0 && M.y.nValid > 0, '缺少獨立驗證段，不能以 training fit 代替驗證。');
disp('X 軸模型 [Kv, tau, L]：');
disp([M.x.Kv, M.x.tau, M.x.L]);
disp('Y 軸模型 [Kv, tau, L]：');
disp([M.y.Kv, M.y.tau, M.y.L]);
disp('X 軸 P1DI 保留驗證 fit：');
disp(M.x.fitValidP);
disp('Y 軸 P1DI 保留驗證 fit：');
disp(M.y.fitValidP);
```

結果存為該 run 裡的 matlab_sysid.mat、matlab_sysid.png，並在命令視窗輸出兩種模型的比較。第二個參數 false 會輸出分析檔案，但不覆寫正式 config/model.json。

檢查保留驗證曲線能否跟上實際位置、fit 是否一致、參數是否卡在上下限，並進一步做殘差與延遲診斷。前面新增的 analyze_sim_sysid 僅供模擬、有 simulationOnly 檢查，不要用來分析真機；真機使用這裡的原專案 sysid_tello。

## 9. 保存資料、恢復連網

- 保留整個 run，不要只傳影片或截圖。核心包含 log.csv、meta.json、scene_frame.jpg、camera_half.mp4，以及估算後的 MAT / PNG。
- 這條 webcam 路線會嘗試自動保存半解析度相機影片。辨識以 log.csv 的時間欄位為準，不要以影片播放時間推回控制時間。
- 記下電池、攝影機位置、距牆距離、光線、異常或人工中止情況。
- 飛機已落地、程式已結束後切回一般 Wi-Fi，再把 run 資料夾交給組員或讓我檢查。
- 目前沒有寫回正式模型或調 PID。這不影響完成資料收集與初步估算。

## 10. 驗證通過後才選做：更新正式模型與 PID

本次第一次真機建議先保留第 8 步結果，重新連網後一起檢查。若你和組員已確認模型與獨立驗證可信，以下是後續指令，會改寫正式設定：

```matlab
backupDir = fullfile(projectRoot, 'runs', ...
    ['config_backup_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))]);
mkdir(backupDir);
copyfile(fullfile(projectRoot, 'config', 'model.json'), fullfile(backupDir, 'model.json'));
copyfile(fullfile(projectRoot, 'config', 'gains.json'), fullfile(backupDir, 'gains.json'));
M = sysid_tello(runDir, true);   % 覆寫正式模型
design_pid([], 60, true);        % 依新模型設計並覆寫正式 PID
```

這段不會自行起飛。不要把計算出 PID 當作已完成真機控制驗證；下一次飛行仍需分階段確認。

## 常見問題

| 狀況 | 處理 |
|---|---|
| webcamlist 為空或 snapshot 失敗 | 檢查 USB、相機權限、其他占用程式；先解決，不起飛 |
| 預覽不是正確鏡頭 | q 退出，重新確認鏡頭與 camera.index |
| scene calibration failed | 板型、尺寸與 'circle' 設定一致，整張板入鏡，改善光線與對比 |
| Tello did not answer command | 確認仍連 TELLO Wi-Fi，關閉其他控制程式；檢查通訊埠／防火牆 |
| Address already in use | 檢查物件未關閉或其他程式占用 UDP；先清理／重啟 MATLAB |
| 電量 NaN | 電量查詢失敗，不視為通過；重新檢查連線 |
| drone not found / GOTO timeout / left safe area | 該次不是正常完成，保留資料並檢查相機、追蹤與方向 |
| 沒有 EXCITE 或沒有驗證段 | 先不要寫回模型，檢查是否選錯 run 或實驗未完整 |

現場安全中止優先於取得完整資料。所有電腦指令都依賴連線與設備正常，不代表保證無人機一定執行。


## 已完成流程不等於資料品質通過

2026-11-10 的 174033、174633 均以 finished 結束，但 X 大振幅測試被範圍保護截斷。未 finished 的資料也可能保留完整測試段，因此第 7 步的 assert 是首次操作的完整性提醒，不是「所有部分資料一律不能分析」的規則。由有經驗的人逐段核對後，可將可靠片段留作探索或額外驗證。

PRBS 的振幅目前在 run_mission.m 寫死為 30；只改 amps 不會改變 PRBS。設定中的 simWind 只作用於模擬，對真機沒有消除風擾的效果。參數調整尚需現場驗證，保留範圍保護。

`audit_real_sysid` 可重現本次四份資料的品質與跨次檢查，不起飛、不改模型；它不重新估算動態參數，且不能自動證明影像追蹤正確。指令見交接報告。
