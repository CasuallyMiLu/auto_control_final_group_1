"""Run deployed YOLO + fusion tracker over the final-project videos; per-frame CSV + contact sheet of detections."""
import sys, json, csv, glob, os, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.expanduser('~/ml'))
from dronetrack.fusion import Associator, Detector, OnlineBackground, measure
md = Path(os.path.expanduser('~/ml/work/deploy/y26s_det_960'))
tc = json.loads((md/'tracker.json').read_text()); fc = tc['fusion']
out = Path(os.path.expanduser('~/ml/final_probe')); out.mkdir(exist_ok=True)
det = Detector(str(md/'best.pt'), tc['imgsz'], conf=0.05, half=True, device=0)
for v in sorted(glob.glob(os.path.expanduser('~/ml/videos_final/*.mp4'))):
    name = Path(v).stem
    cap = cv2.VideoCapture(v); fps = cap.get(5)
    bgm = OnlineBackground(fc['bg_scale'], fc['bg_warmup'], fc['bg_alpha'], fc.get('motion_px', 1.0))
    asc = Associator(mode=tc['mode'], conf=tc['conf'], conf_reacquire=max(tc['conf'], fc['conf_reacquire']), gate_px=fc['gate_px'],
                     coast=fc['coast'], kalman=tc['kalman'], q=tc['q'], r=fc['kalman_r'], static_s=0.0, static_px=1.5)
    rows=[]; i=0; t0=time.time(); crops=[]
    while True:
        ok, f = cap.read()
        if not ok: break
        dets, _ = measure(f, det, bgm, fc)
        o = asc.step(dets, i/fps)
        best = max(dets, key=lambda d: d['conf']) if dets else None
        bw = bh = cf = np.nan
        if best is not None:
            x0,y0,x1,y1 = best['box']; bw, bh, cf = x1-x0, y1-y0, best['conf']
            if i % 40 == 0 and len(crops) < 24:
                cx, cy = int((x0+x1)/2), int((y0+y1)/2)
                c = f[max(0,cy-80):cy+80, max(0,cx-80):cx+80]
                c = cv2.copyMakeBorder(c,0,160-c.shape[0],0,160-c.shape[1],cv2.BORDER_CONSTANT)
                cv2.putText(c, f"{i} {cf:.2f}", (2,14), 0, .45, (0,255,255), 1); crops.append(c)
        rows.append([i, i/fps, o['x'], o['y'], len(dets), cf, bw, bh])
        i += 1
    with open(out/f'{name}.csv','w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['frame','t','u','v','ndet','conf','bw','bh']); w.writerows(rows)
    a = np.array(rows, float)
    print(name, 'frames', i, 'fps', round(fps,2), 'valid%', round(100*np.isfinite(a[:,2]).mean(),1), 'det%', round(100*(a[:,4]>0).mean(),1),
          'med box', np.nanmedian(a[:,6]).round(1), np.nanmedian(a[:,7]).round(1), 'proc fps', round(i/(time.time()-t0),1), flush=True)
    if crops:
        while len(crops)%6: crops.append(np.zeros_like(crops[0]))
        cv2.imwrite(str(out/f'{name}_crops.jpg'), np.vstack([np.hstack(crops[k:k+6]) for k in range(0,len(crops),6)]))
