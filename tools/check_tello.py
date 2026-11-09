"""No-flight check: connect to the Tello, print battery and 3 s of the state stream. Motors stay off.
python tools/check_tello.py      (laptop Wi-Fi on TELLO-xxxxxx; MATLAB ryze object closed)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ftrack.tello import Tello
t = Tello()
if not t.connect():
    sys.exit("no answer to 'command': check Wi-Fi (TELLO-xxxxxx), firewall (allow python UDP 8889/8890), MATLAB closed")
print("connected; battery", t.battery(), "%")
t0 = time.time()
while time.time() - t0 < 3:
    time.sleep(0.5)
    st = t.state
    print({k: st.get(k) for k in ("bat", "h", "tof", "vgx", "vgy", "vgz", "pitch", "roll", "yaw", "templ", "temph")})
if not t.state:
    print("WARNING: no state packets on UDP 8890 (firewall?) - flights still work, but the log has no Tello telemetry")
t.close()
