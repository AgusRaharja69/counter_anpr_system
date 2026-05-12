"""
1_detect.py  —  Raspberry Pi 5  (Windows-compatible untuk testing)
====================================================================
Deteksi kendaraan, ROI crossing, rekam clip + thumbnail, kirim ke VPS.

PERBAIKAN v3:
  - Folder captures/ dibuat relatif ke lokasi script (bukan path absolut)
  - Clip direkam dengan anotasi: ROI line, bbox, counter, timestamp
  - Upload via paramiko SFTP (tidak butuh rsync/sshpass di Windows)
  - rsync tetap tersedia sebagai opsional jika ada di sistem
  - MQTT fix DeprecationWarning callback API
  - MQTT payload status=0 default, plate_number=None jika ANPR off
  - Hitung kendaraan MASUK SAJA (atas → bawah ROI)
  - OCR: 3_anpr_ocr_custom.py (ONNX → PT → EasyOCR fallback)

Usage:
  python 1_detect.py --source rtsp
  python 1_detect.py --source video --file test.mp4
  python 1_detect.py --source folder --file videos/
  python 1_detect.py --setup-roi --source folder --file videos/
  python 1_detect.py --source folder --file videos/ --anpr

Install:
  pip install ultralytics opencv-python paho-mqtt paramiko numpy
  (opsional) pip install easyocr onnxruntime
"""

import cv2, json, os, re, sys, time, base64, argparse, threading
import subprocess
from datetime import datetime
from pathlib import Path
from collections import deque
from configparser import ConfigParser

# ─── BASE DIR (lokasi script, bukan cwd) ──────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ─── LOAD CONFIG.INI ──────────────────────────────────────────────────────────
def load_config(path='config.ini'):
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    ini = ConfigParser()
    if not p.exists():
        print(f'[ERROR] config.ini tidak ditemukan: {p}')
        raise SystemExit(1)
    ini.read(str(p), encoding='utf-8')

    def get(section, key, fallback=None):
        v = ini.get(section, key, fallback=fallback or '')
        return v.strip() if v else (fallback or '')

    def getbool(section, key, fallback=False):
        return get(section, key, str(fallback)).lower() in ('true','1','yes')

    def getfloat(section, key, fallback=0.0):
        try:    return float(get(section, key, str(fallback)))
        except: return fallback

    def getint(section, key, fallback=0):
        try:    return int(get(section, key, str(fallback)))
        except: return fallback

    # LocalPath: jika relatif, jadikan relatif ke BASE_DIR
    raw_local = get('VIDEO', 'LocalPath', fallback='captures').strip('/')
    if os.path.isabs(raw_local):
        local_path = raw_local
    else:
        local_path = str(BASE_DIR / raw_local)

    cfg = {
        # MQTT
        'mqtt_broker'     : get('DEFAULT', 'Broker',  fallback=''),
        'mqtt_port'       : getint('DEFAULT', 'Port', fallback=1883),
        'mqtt_user'       : get('DEFAULT', 'UserID',  fallback=''),
        'mqtt_pass'       : get('DEFAULT', 'Pass',    fallback=''),
        'mqtt_prefix'     : 'anpr',

        # VIDEO / LOCAL
        'clip_sec_before' : getfloat('VIDEO', 'ClipBefore',     fallback=2),
        'clip_sec_after'  : getfloat('VIDEO', 'ClipAfter',      fallback=3),
        'clip_fps'        : getint('VIDEO',   'ClipFPS',        fallback=15),
        'thumbnail_w'     : getint('VIDEO',   'ThumbnailWidth', fallback=320),
        'local_clips'     : local_path + '/clips',
        'local_thumbs'    : local_path + '/thumbnails',
        'local_events'    : local_path + '/events',

        # VPS / SFTP
        'vps_host'        : get('VIDEO', 'RemoteHost', fallback=''),
        'vps_user'        : get('VIDEO', 'RemoteUser', fallback=''),
        'vps_pass'        : get('VIDEO', 'RemotePass', fallback=''),
        'vps_base'        : get('VIDEO', 'RemotePath', fallback='/home/anpr'),
        'vps_ssh_key'     : get('VIDEO', 'RemoteKey',  fallback=''),
        'upload_every_s'  : getint('VIDEO',   'RsyncInterval',        fallback=120),
        'upload_delete'   : getbool('VIDEO',  'RsyncDeleteAfterSend', fallback=True),

        # CAMERA
        'source'          : get('CAMERA', 'Source',        fallback='rtsp'),
        'rtsp_url'        : get('CAMERA', 'RTSP',          fallback=''),
        'file_path'       : get('CAMERA', 'FilePath',      fallback=''),
        'reconnect_s'     : getint('CAMERA', 'ReconnectDelay', fallback=5),
        'process_every'   : getint('CAMERA', 'ProcessEvery',   fallback=2),

        # DEVICE
        'device_id'       : get('DEVICE', 'DeviceID', fallback='gate-01'),
        'device_location' : get('DEVICE', 'Location', fallback=''),

        # ROI
        'roi_y'           : getfloat('ROI', 'Y',  fallback=0.55),
        'roi_x1'          : getfloat('ROI', 'X1', fallback=0.05),
        'roi_x2'          : getfloat('ROI', 'X2', fallback=0.95),

        # MODEL
        'vehicle_model'   : get('MODEL', 'VehicleModel',     fallback='models/yolov8n.pt'),
        'vehicle_conf'    : getfloat('MODEL', 'VehicleConf', fallback=0.45),
        'plate_model'     : get('MODEL', 'PlateModel',       fallback='models/plate_detector.pt'),
        'plate_conf'      : getfloat('MODEL', 'PlateConf',   fallback=0.30),
        'vehicle_classes' : [2, 3, 5, 7],

        # ANPR
        'anpr_enabled'    : getbool('ANPR', 'Enabled',  fallback=False),
        'anpr_onnx'       : get('ANPR', 'OnnxModel',    fallback='models/plate_ocr.onnx'),
        'anpr_pt'         : get('ANPR', 'PtModel',      fallback='models/plate_ocr.pt'),
        'anpr_class_map'  : get('ANPR', 'ClassMap',     fallback='models/class_mapping.json'),

        # TRACKER
        'tracker_iou'     : getfloat('TRACKER', 'IOU',             fallback=0.45),
        'tracker_max_age' : getint('TRACKER',   'MaxAge',          fallback=25),
        'cooldown_s'      : getfloat('TRACKER', 'CooldownSeconds', fallback=8.0),

        # DISPLAY
        'display_width'   : getint('CAMERA', 'DisplayWidth', fallback=1280),
    }

    # Resolve model path relatif ke BASE_DIR
    for k in ('vehicle_model','plate_model','anpr_onnx','anpr_pt','anpr_class_map'):
        v = cfg[k]
        if v and not os.path.isabs(v):
            cfg[k] = str(BASE_DIR / v)

    cfg['rsync_use_key'] = bool(cfg['vps_ssh_key'])
    return cfg


ROI_FILE = str(BASE_DIR / 'roi_config.json')
VLABEL   = {2:'Car', 3:'Motorcycle', 5:'Bus', 7:'Truck'}
CFG      = {}


# ─── ROI ──────────────────────────────────────────────────────────────────────
class ROI:
    def __init__(self, fw, fh):
        self.fw, self.fh = fw, fh
        d = json.load(open(ROI_FILE)) if os.path.exists(ROI_FILE) else {}
        self.y  = int(d.get('roi_y',  CFG['roi_y'])  * fh)
        self.x1 = int(d.get('roi_x1', CFG['roi_x1']) * fw)
        self.x2 = int(d.get('roi_x2', CFG['roi_x2']) * fw)

    def draw(self, frame, cnt_total=0):
        """Gambar garis ROI + label counter + arah masuk."""
        # Garis ROI
        cv2.line(frame, (self.x1, self.y), (self.x2, self.y), (0,255,0), 2)
        # Label kiri
        cv2.putText(frame, f'ROI  v MASUK: {cnt_total}',
                    (self.x1+4, self.y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Panah tengah
        mid = (self.x1 + self.x2) // 2
        cv2.arrowedLine(frame, (mid, self.y-30), (mid, self.y+8),
                        (0,255,128), 2, tipLength=0.4)
        return frame

    def crossed_downward(self, prev_y, cur_y):
        """True hanya jika bergerak atas→bawah (y bertambah) melewati ROI."""
        return prev_y < self.y <= cur_y

    def near(self, y, m=35):
        return abs(y - self.y) < m


def setup_roi(frame):
    pts = []
    def cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
    cv2.namedWindow('ROI Setup', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('ROI Setup', cb)
    print('Klik 2 titik -> Enter/S simpan | R reset | Q batal')
    print('Kendaraan dari ATAS ke BAWAH garis = MASUK')
    h, w = frame.shape[:2]
    while True:
        d = frame.copy()
        for p in pts:
            cv2.circle(d, p, 6, (0,0,255), -1)
        if len(pts) == 2:
            cv2.line(d, pts[0], pts[1], (0,255,0), 2)
        cv2.putText(d, 'ROI Setup - arah masuk: atas ke bawah | Enter=simpan',
                    (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow('ROI Setup', d)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, ord('s')) and len(pts) == 2:
            cfg = {
                'roi_y'  : (pts[0][1]+pts[1][1])/2/h,
                'roi_x1' : min(pts[0][0],pts[1][0])/w,
                'roi_x2' : max(pts[0][0],pts[1][0])/w,
            }
            json.dump(cfg, open(ROI_FILE,'w'), indent=2)
            print(f'ROI saved: {cfg}')
            break
        elif k == ord('r'): pts.clear()
        elif k == ord('q'): break
    cv2.destroyAllWindows()


# ─── TRACKER ──────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.tracks = {}
        self.nid    = 0

    @staticmethod
    def iou(a, b):
        ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
        ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        ua = (a[2]-a[0])*(a[3]-a[1])
        ub = (b[2]-b[0])*(b[3]-b[1])
        return inter/(ua+ub-inter+1e-6)

    def update(self, dets):
        for t in self.tracks.values():
            t['age'] += 1
        self.tracks = {k:v for k,v in self.tracks.items()
                       if v['age'] <= CFG['tracker_max_age']}
        matched = set()
        results = []
        for det in dets:
            x1,y1,x2,y2,cls,conf = det
            best_sc, best_tid = CFG['tracker_iou'], None
            for tid, trk in self.tracks.items():
                if tid in matched: continue
                sc = self.iou((x1,y1,x2,y2), trk['bbox'])
                if sc > best_sc: best_sc,best_tid = sc,tid
            prev_y2 = y2
            if best_tid is not None:
                prev_y2 = self.tracks[best_tid]['bbox'][3]
                self.tracks[best_tid].update(
                    {'bbox':(x1,y1,x2,y2),'age':0,'cls':cls,'conf':conf})
                matched.add(best_tid)
                results.append((x1,y1,x2,y2,cls,conf,best_tid,prev_y2))
            else:
                tid = self.nid; self.nid += 1
                self.tracks[tid] = {
                    'bbox':(x1,y1,x2,y2),'age':0,
                    'crossed':False,'cls':cls,'conf':conf}
                results.append((x1,y1,x2,y2,cls,conf,tid,y2))
        return results

    def mark(self, tid):
        if tid in self.tracks: self.tracks[tid]['crossed'] = True

    def has_crossed(self, tid):
        return self.tracks.get(tid,{}).get('crossed', False)


# ─── CLIP BUFFER (simpan frame ter-anotasi) ───────────────────────────────────
class ClipBuffer:
    """
    Buffer frame TER-ANOTASI (sudah ada ROI line, bbox, counter).
    Clip yang direkam = persis seperti yang tampil di layar.
    """
    def __init__(self, fps):
        n = int(CFG['clip_sec_before'] * fps)
        self.buf = deque(maxlen=n)   # frame ter-anotasi
        self.fps = fps
        self._writers = {}

    def push(self, annotated_frame):
        """Push frame yang sudah ter-anotasi."""
        self.buf.append(annotated_frame.copy())

    def start(self, eid, shape, path):
        h,w = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        wr = cv2.VideoWriter(path, fourcc, self.fps, (w,h))
        for f in self.buf:
            wr.write(f)
        max_n = int(CFG['clip_sec_after'] * self.fps)
        self._writers[eid] = {'wr':wr,'n':0,'max':max_n}

    def feed(self, annotated_frame):
        done = []
        for eid,d in self._writers.items():
            d['wr'].write(annotated_frame)
            d['n'] += 1
            if d['n'] >= d['max']:
                d['wr'].release(); done.append(eid)
        for e in done: del self._writers[e]


# ─── SFTP UPLOADER (paramiko, tidak butuh rsync/sshpass) ─────────────────────
class SFTPUploader:
    """
    Upload file ke VPS via SFTP (paramiko).
    Fallback ke rsync jika tersedia di sistem.
    Berjalan di background thread tiap N detik.
    """
    def __init__(self):
        self._has_paramiko = self._check_paramiko()
        self._has_rsync    = self._check_rsync()
        threading.Thread(target=self._loop, daemon=True).start()

    def _check_paramiko(self):
        try:
            import paramiko; return True
        except ImportError:
            print('  [INFO] paramiko tidak ada. Install: pip install paramiko')
            return False

    def _check_rsync(self):
        try:
            r = subprocess.run(['rsync','--version'],
                               capture_output=True, timeout=5)
            if r.returncode == 0:
                print('  upload: rsync tersedia')
                return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        return False

    def _has_sshpass(self):
        try:
            r = subprocess.run(['sshpass','-V'],
                               capture_output=True, timeout=3)
            return r.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    def _remote_dir(self, sub):
        """Path folder tujuan di VPS."""
        return f"{CFG['vps_base']}/{CFG['device_id']}/{sub}"

    # ── SFTP via paramiko ────────────────────────────────────────────
    @staticmethod
    def _sftp_makedirs(sftp, remote_path):
        # Buat folder rekursif via SFTP (seperti mkdir -p)
        dirs = []
        path = remote_path
        while True:
            try:
                sftp.stat(path)
                break
            except IOError:
                dirs.append(path)
                parent = path.rsplit('/', 1)[0]
                if not parent or parent == path:
                    break
                path = parent
        for d in reversed(dirs):
            try:
                sftp.mkdir(d)
            except IOError:
                pass

    def _sftp_connect(self):
        # Buka koneksi SSH+SFTP, return (ssh, sftp) atau raise
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key_path = CFG['vps_ssh_key'] or os.path.expanduser('~/.ssh/id_rsa')
        try:
            if os.path.exists(key_path):
                ssh.connect(CFG['vps_host'], username=CFG['vps_user'],
                            key_filename=key_path, timeout=20)
                print('  SFTP: connected (key)')
            else:
                raise FileNotFoundError
        except Exception:
            ssh.connect(CFG['vps_host'], username=CFG['vps_user'],
                        password=CFG['vps_pass'], timeout=20)
            print('  SFTP: connected (password)')
        return ssh, ssh.open_sftp()

    def _sftp_upload_dir(self, local_dir, sub):
        local = Path(local_dir)
        files = [f for f in local.glob('*') if f.is_file()]
        if not files:
            return
        try:
            ssh, sftp = self._sftp_connect()
            remote_dir = self._remote_dir(sub)

            # Buat folder tujuan rekursif sepenuhnya via SFTP
            self._sftp_makedirs(sftp, remote_dir)

            uploaded = []
            for f in files:
                remote_path = f"{remote_dir}/{f.name}"
                try:
                    sftp.put(str(f), remote_path)
                    uploaded.append(f)
                    print(f'  SFTP {sub}: OK {f.name}')
                except Exception as upload_err:
                    print(f'  SFTP {sub}: SKIP {f.name} ({upload_err})')

            sftp.close()
            ssh.close()

            if CFG['upload_delete']:
                for f in uploaded:
                    try: f.unlink()
                    except: pass

        except Exception as e:
            print(f'  SFTP {sub}: ERR {e}')

    # ── rsync (jika ada di sistem) ─────────────────────────────────────────────
    def _rsync_upload_dir(self, local_dir, sub):
        if not os.path.isdir(local_dir): return
        base = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
        if CFG['rsync_use_key']:
            ssh_cmd = f"{base} -i {CFG['vps_ssh_key']}"
        else:
            ssh_cmd = f"sshpass -p '{CFG['vps_pass']}' {base}"
        dest = (f"{CFG['vps_user']}@{CFG['vps_host']}:"
                f"{CFG['vps_base']}/{CFG['device_id']}/{sub}/")
        cmd = ['rsync','-avz','--mkpath','-e', ssh_cmd]
        if CFG['upload_delete']: cmd.append('--remove-source-files')
        cmd += [local_dir+'/', dest]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=90)
            if r.returncode == 0: print(f'  rsync {sub}: OK')
            else: print(f'  rsync {sub}: ERR {r.stderr.decode()[:150]}')
        except subprocess.TimeoutExpired:
            print(f'  rsync {sub}: TIMEOUT')
        except Exception as e:
            print(f'  rsync {sub}: exception {e}')

    def _upload_all(self, label='auto'):
        if not CFG['vps_host']:
            return
        pairs = [
            (CFG['local_clips'],  'clips'),
            (CFG['local_thumbs'], 'thumbnails'),
            (CFG['local_events'], 'events'),
        ]
        # Hitung total file
        total_files = sum(
            len([f for f in Path(local).glob('*') if f.is_file()])
            for local, _ in pairs if os.path.isdir(local))

        ts = datetime.now().strftime('%H:%M:%S')
        if total_files == 0:
            print(f'  [{ts}] Upload ({label}): tidak ada file baru')
            return

        print(f'  [{ts}] Upload ({label}): {total_files} file → {CFG["vps_host"]}')

        for local, sub in pairs:
            if not os.path.isdir(local): continue
            n = len([f for f in Path(local).glob('*') if f.is_file()])
            if n == 0: continue
            if self._has_paramiko and CFG['vps_pass']:
                self._sftp_upload_dir(local, sub)
            elif self._has_rsync and CFG['rsync_use_key']:
                self._rsync_upload_dir(local, sub)
            elif self._has_rsync and self._has_sshpass():
                self._rsync_upload_dir(local, sub)
            elif self._has_paramiko:
                self._sftp_upload_dir(local, sub)
            else:
                print(f'  [WARN] Tidak ada cara upload. Install: pip install paramiko')
                break

    def _loop(self):
        next_up = time.time() + CFG['upload_every_s']
        while True:
            now = time.time()
            if now >= next_up:
                self._upload_all(label='tiap 2mnt')
                next_up = time.time() + CFG['upload_every_s']
            time.sleep(5)

    def upload_now(self):
        # Upload segera secara blocking
        self._upload_all(label='manual')


# ─── MQTT ─────────────────────────────────────────────────────────────────────
class MQTTClient:
    def __init__(self):
        self._ok = False
        try:
            import paho.mqtt.client as mqtt
            # paho-mqtt >= 2.0: gunakan VERSION2 untuk hilangkan DeprecationWarning
            try:
                self.c = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"rpi-{CFG['device_id']}-{int(time.time())}")
                # VERSION2: callback on_connect(c,u,f,rc,props), on_disconnect(c,u,disc,props)
                self.c.on_connect    = lambda c,u,f,rc,p: self._conn(rc)
                self.c.on_disconnect = lambda c,u,d,p: print(f'  MQTT disc rc={d.rc if hasattr(d,"rc") else d}')
            except AttributeError:
                # paho-mqtt < 2.0
                self.c = mqtt.Client(
                    client_id=f"rpi-{CFG['device_id']}-{int(time.time())}")
                self.c.on_connect    = lambda c,u,f,rc: self._conn(rc)
                self.c.on_disconnect = lambda c,u,rc: print(f'  MQTT disc rc={rc}')

            if CFG['mqtt_user']:
                self.c.username_pw_set(CFG['mqtt_user'], CFG['mqtt_pass'])
            self.c.will_set(
                self._t('status'),
                json.dumps({'device_id':CFG['device_id'],'online':False,'status':0}),
                qos=1, retain=True)
            self.c.connect_async(CFG['mqtt_broker'], CFG['mqtt_port'], 60)
            self.c.loop_start()
            self._ok = True
        except Exception as e:
            print(f'  MQTT init failed: {e}')

    def _t(self, s):
        return f"{CFG['mqtt_prefix']}/{CFG['device_id']}/{s}"

    def _conn(self, rc):
        if rc == 0:
            print(f"  MQTT connected -> {CFG['mqtt_broker']}")
            self.pub('status', {
                'device_id'   : CFG['device_id'],
                'location'    : CFG['device_location'],
                'online'      : True,
                'status'      : 0,   # default status
                'anpr_enabled': CFG['anpr_enabled'],
                'ts'          : datetime.now().isoformat(),
            }, retain=True)

    def pub(self, sfx, data, qos=1, retain=False):
        if not self._ok: return
        try:
            self.c.publish(self._t(sfx),
                json.dumps(data, default=str), qos=qos, retain=retain)
        except Exception as e:
            print(f'  MQTT pub error: {e}')


# ─── ANPR ─────────────────────────────────────────────────────────────────────
class ANPR:
    """
    Wrapper PlateOCR dari 3_anpr_ocr_custom.py.
    Prioritas: ONNX → PyTorch (.pt) → EasyOCR fallback.
    """
    def __init__(self):
        self._ocr    = None
        self._reader = None
        self._load()

    def _load(self):
        ocr_script = BASE_DIR / '3_anpr_ocr_custom.py'
        if ocr_script.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    'anpr_ocr', str(ocr_script))
                mod = importlib.util.module_from_spec(spec)
                sys.modules['anpr_ocr'] = mod
                spec.loader.exec_module(mod)
                mod.CONFIG['model_onnx']  = CFG['anpr_onnx']
                mod.CONFIG['model_pt']    = CFG['anpr_pt']
                mod.CONFIG['class_map']   = CFG['anpr_class_map']
                mod.CONFIG['plate_model'] = CFG['plate_model']
                mod.CONFIG['plate_conf']  = CFG['plate_conf']
                self._ocr = mod.PlateOCR()
                print('  ANPR: Custom CNN OCR ready')
                return
            except Exception as e:
                print(f'  ANPR: custom OCR gagal ({e}), coba EasyOCR...')
        try:
            import easyocr
            self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print('  ANPR: EasyOCR fallback ready')
        except Exception as e:
            print(f'  ANPR init gagal: {e}')

    def run(self, img):
        if img is None: return None, 0.0
        if self._ocr is not None:
            try: return self._ocr.recognize_plate(img)
            except Exception as e: print(f'  ANPR OCR error: {e}')
        if self._reader is not None:
            return self._easyocr(img)
        return None, 0.0

    def _easyocr(self, img):
        try:
            res = self._reader.readtext(img, detail=1, paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if not res: return None, 0.0
            raw  = re.sub(r'[^A-Z0-9]','',''.join(r[1] for r in res).upper())
            conf = sum(r[2] for r in res)/len(res)
            return self._correct(raw), conf
        except: return None, 0.0

    @staticmethod
    def _correct(text):
        A2D = {'O':'0','Q':'0','I':'1','L':'1','Z':'9','S':'5','G':'6','T':'7','B':'8','P':'9'}
        D2A = {'0':'O','1':'I','5':'S','6':'G','7':'T','8':'B'}
        text = re.sub(r'[^A-Z0-9]','', text.upper())
        if len(text) < 5: return None
        for pl in [1,2]:
            for sl in [1,2,3]:
                nl = len(text)-pl-sl
                if not 1<=nl<=4: continue
                rp = ''.join(D2A.get(c,c) for c in text[:pl])
                rn = ''.join(A2D.get(c,c) if not c.isdigit() else c for c in text[pl:pl+nl])
                rs = ''.join(D2A.get(c,c) for c in text[pl+nl:])
                if rp.isalpha() and rn.isdigit() and rs.isalpha():
                    return f"{rp} {rn} {rs}"
        return None


# ─── DETECTOR ─────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, source=None, file_path=None, anpr=False):
        self.source   = source    or CFG['source']
        self.fp       = file_path or CFG['file_path']
        self.tracker  = Tracker()
        self.uploader = SFTPUploader()
        self.mqtt     = MQTTClient()
        self.anpr     = ANPR() if anpr else None
        self._cd      = {}
        self._fn      = 0
        self.roi      = None
        self.buf      = None
        self.cnt      = {'total':0,'car':0,'motorcycle':0,'bus':0,'truck':0}

        # Buat folder lokal
        for d in [CFG['local_clips'], CFG['local_thumbs'], CFG['local_events']]:
            Path(d).mkdir(parents=True, exist_ok=True)
        print(f'  Folder lokal: {Path(CFG["local_clips"]).parent}')

        print('Loading YOLOv8n...')
        # Fix torch >= 2.6: weights_only default berubah jadi True
        # patch agar YOLO model (.pt) bisa di-load tanpa error
        try:
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    from ultralytics.nn.tasks import (
                        DetectionModel, SegmentationModel,
                        ClassificationModel, PoseModel)
                    torch.serialization.add_safe_globals([
                        DetectionModel, SegmentationModel,
                        ClassificationModel, PoseModel])
                except ImportError:
                    pass
        except Exception:
            pass

        from ultralytics import YOLO
        self.vm = YOLO(CFG['vehicle_model'])
        self.pm = None
        if os.path.exists(CFG['plate_model']):
            self.pm = YOLO(CFG['plate_model'])
            print('  Plate detector loaded')

    # ── Buka sumber video ──────────────────────────────────────────────────────
    def _open(self):
        if self.source == 'rtsp':
            # Paksa TCP + no-buffer SEBELUM buka capture
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'fflags;nobuffer|'
                'flags;low_delay|'
                'allowed_media_types;video')
            cap = cv2.VideoCapture(CFG['rtsp_url'], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Turunkan resolusi baca jika kamera 4K/2K (hemat CPU RPi)
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                print('  [WARN] Gagal buka RTSP')
            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f'  RTSP OK: {w}x{h}')
            return cap
        elif self.source == 'folder':
            fp = Path(self.fp) if os.path.isabs(self.fp) else BASE_DIR / self.fp
            files = sorted(fp.glob('*.mp4')) + sorted(fp.glob('*.avi'))
            if not files:
                print(f'  [WARN] Tidak ada video di: {fp}')
            self._fi = iter(files)
            return self._nf()
        elif self.source == 'webcam':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            fp = Path(self.fp) if os.path.isabs(self.fp) else BASE_DIR / self.fp
            cap = cv2.VideoCapture(str(fp))
            return cap

    def _nf(self):
        p = next(self._fi)
        print(f'  Playing: {p}')
        return cv2.VideoCapture(str(p))

    # ── Deteksi YOLO ──────────────────────────────────────────────────────────
    def _detect_v(self, frame):
        res = self.vm(frame, conf=CFG['vehicle_conf'],
                      classes=CFG['vehicle_classes'], verbose=False)
        out = []
        for r in res:
            for b in r.boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                out.append((x1,y1,x2,y2,int(b.cls[0]),float(b.conf[0])))
        return out

    def _detect_plate(self, vcrop):
        if not self.pm or vcrop is None: return None
        try:
            res = self.pm(vcrop, conf=CFG['plate_conf'], verbose=False)
            best, bc = None, 0
            for r in res:
                for b in r.boxes:
                    c = float(b.conf[0])
                    if c > bc:
                        bc = c; best = b.xyxy[0].cpu().numpy().astype(int)
            if best is not None:
                x1,y1,x2,y2 = best; m=6; ph,pw = vcrop.shape[:2]
                return vcrop[max(0,y1-m):min(ph,y2+m), max(0,x1-m):min(pw,x2+m)]
        except: pass
        return None

    # ── Anotasi frame ─────────────────────────────────────────────────────────
    def _annotate(self, frame, tracked):
        """
        Gambar ROI, bbox kendaraan, counter, timestamp ke frame.
        Frame yang sudah ter-anotasi ini yang masuk ke ClipBuffer dan ditampilkan.
        """
        # ROI line + counter
        if self.roi:
            self.roi.draw(frame, self.cnt['total'])

        # Bbox per kendaraan
        for det in tracked:
            x1,y1,x2,y2,cls,conf,tid,prev_y2 = det
            moving_down = y2 > prev_y2
            if self.roi and self.roi.near(y2):
                col = (0,220,100) if moving_down else (80,80,200)
            else:
                col = (0,160,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            arrow = 'v' if moving_down else '^'
            label = f"{VLABEL.get(cls,'?')}#{tid} {arrow}"
            cv2.putText(frame, label, (x1, max(y1-6,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # Timestamp
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, ts, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        return frame

    # ── Event crossing ────────────────────────────────────────────────────────
    def _on_cross(self, annotated_frame, raw_frame, det):
        """
        annotated_frame : frame dengan ROI + bbox (untuk thumbnail & clip)
        raw_frame       : frame asli (untuk crop kendaraan ke ANPR)
        """
        x1,y1,x2,y2,cls,conf,tid,_ = det
        vtype = VLABEL.get(cls,'Unknown')

        # Cooldown anti-duplikat
        ck  = f"{cls}_{x1//80}_{y1//80}"
        now = time.time()
        if now - self._cd.get(ck,0) < CFG['cooldown_s']:
            self.tracker.mark(tid); return
        self._cd[ck] = now
        self.tracker.mark(tid)

        # Counter
        self.cnt['total'] += 1
        self.cnt[vtype.lower()] = self.cnt.get(vtype.lower(),0) + 1

        ts  = datetime.now()
        eid = f"{CFG['device_id']}_{ts.strftime('%Y%m%d_%H%M%S')}_{tid:04d}"
        print(f"  [{ts.strftime('%H:%M:%S')}] {vtype} MASUK -> {eid}")

        fh,fw = annotated_frame.shape[:2]
        m = 16

        # ── Thumbnail dari frame TER-ANOTASI ──────────────────────────────────
        crop  = annotated_frame[max(0,y1-m):min(fh,y2+m),
                                max(0,x1-m):min(fw,x2+m)]
        tw    = CFG['thumbnail_w']
        th    = int(crop.shape[0]*tw/max(crop.shape[1],1))
        if th > 0 and tw > 0:
            thumb = cv2.resize(crop,(tw,th))
        else:
            thumb = crop
        tpath = os.path.join(CFG['local_thumbs'], f"{eid}.jpg")
        cv2.imwrite(tpath, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # ── Clip dari buffer (sudah ter-anotasi) ──────────────────────────────
        cpath = os.path.join(CFG['local_clips'], f"{eid}.mp4")
        if self.buf:
            self.buf.start(eid, annotated_frame.shape, cpath)

        # ── Crop kendaraan dari RAW frame untuk ANPR (tidak ada overlay) ──────
        vcrop = raw_frame[max(0,y1-16):min(fh,y2+16),
                          max(0,x1-16):min(fw,x2+16)].copy()
        pcrop = self._detect_plate(vcrop)

        # Thumbnail base64 kecil untuk MQTT
        tb64 = ''
        try:
            with open(tpath,'rb') as f:
                tb64 = base64.b64encode(f.read()).decode()
        except: pass

        # ── Event JSON ────────────────────────────────────────────────────────
        event = {
            'event_id'       : eid,
            'device_id'      : CFG['device_id'],
            'device_location': CFG['device_location'],
            'timestamp'      : ts.isoformat(),
            'vehicle_type'   : vtype,
            'direction'      : 'in',
            'confidence'     : round(conf,3),
            'bbox'           : [int(x1),int(y1),int(x2),int(y2)],
            'clip_file'      : f"{eid}.mp4",
            'thumb_file'     : f"{eid}.jpg",
            'plate_number'   : None,     # diisi ANPR async jika aktif
            'plate_conf'     : None,
            'anpr_enabled'   : CFG['anpr_enabled'],
            'status'         : 0,        # default
            'counters'       : dict(self.cnt),
        }

        jpath = os.path.join(CFG['local_events'], f"{eid}.json")
        json.dump(event, open(jpath,'w'), indent=2, default=str)

        # ── Publish MQTT ──────────────────────────────────────────────────────
        mqtt_ev = dict(event)
        mqtt_ev['thumb_b64'] = tb64
        self.mqtt.pub('event', mqtt_ev)
        self.mqtt.pub('counters', {
            'device_id': CFG['device_id'],
            'counters' : self.cnt,
            'status'   : 0,
            'ts'       : ts.isoformat(),
        })

        # ── ANPR async ────────────────────────────────────────────────────────
        if self.anpr and (pcrop is not None or vcrop is not None):
            anpr_input = pcrop if pcrop is not None else vcrop
            def run_anpr(ev, img, jp):
                plate, pconf = self.anpr.run(img)
                if plate:
                    ev['plate_number'] = plate
                    ev['plate_conf']   = round(pconf,3)
                    json.dump(ev, open(jp,'w'), indent=2, default=str)
                    self.mqtt.pub('anpr', {
                        'event_id'    : ev['event_id'],
                        'device_id'   : CFG['device_id'],
                        'plate_number': plate,
                        'plate_conf'  : round(pconf,3),
                        'status'      : 0,
                        'ts'          : datetime.now().isoformat(),
                    })
                    print(f"    ANPR: {plate} ({pconf:.2f})")
            threading.Thread(target=run_anpr,
                args=(event, anpr_input, jpath), daemon=True).start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = self._open()
        self.buf = ClipBuffer(CFG['clip_fps'])

        # Buat window SEKALI, nama tetap sepanjang sesi
        self._win_name = f"ANPR [{CFG['device_id']}]"
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name,
                         CFG.get('display_width', 1280),
                         CFG.get('display_width', 1280) * 9 // 16)

        # Heartbeat tiap 30 detik
        def hb():
            while True:
                time.sleep(30)
                self.mqtt.pub('heartbeat', {
                    'device_id': CFG['device_id'],
                    'location' : CFG['device_location'],
                    'online'   : True,
                    'status'   : 0,
                    'counters' : self.cnt,
                    'ts'       : datetime.now().isoformat(),
                })
        threading.Thread(target=hb, daemon=True).start()

        print(f"\nDetector running [{CFG['device_id']}]. Tekan Q untuk berhenti.\n")

        # Thread khusus baca RTSP agar main loop tidak nunggu network
        self._rtsp_frame  = None
        self._rtsp_ret    = False
        self._rtsp_lock   = threading.Lock()
        self._rtsp_stop   = False

        def rtsp_reader():
            nonlocal cap
            consec_fail = 0
            while not self._rtsp_stop:
                if self.source == 'rtsp':
                    # grab() lebih cepat dari read() — tidak decode dulu
                    ok = cap.grab()
                    if not ok:
                        consec_fail += 1
                        if consec_fail >= 30:
                            print(f"  RTSP putus, reconnect dalam {CFG['reconnect_s']}s...")
                            cap.release()
                            time.sleep(CFG['reconnect_s'])
                            cap = self._open()
                            consec_fail = 0
                        continue
                    consec_fail = 0
                    # retrieve() untuk decode frame yang sudah di-grab
                    ret, frame = cap.retrieve()
                    with self._rtsp_lock:
                        self._rtsp_frame = frame if ret else None
                        self._rtsp_ret   = ret
                else:
                    ret, frame = cap.read()
                    with self._rtsp_lock:
                        self._rtsp_frame = frame if ret else None
                        self._rtsp_ret   = ret

        if self.source == 'rtsp':
            threading.Thread(target=rtsp_reader, daemon=True).start()
            time.sleep(1)  # tunggu frame pertama

        _no_frame_count = 0

        try:
            while True:
                # Ambil frame
                if self.source == 'rtsp':
                    with self._rtsp_lock:
                        ret   = self._rtsp_ret
                        frame = self._rtsp_frame
                    if not ret or frame is None:
                        _no_frame_count += 1
                        if _no_frame_count > 100:
                            print('  Tidak ada frame dari RTSP, tunggu...')
                            _no_frame_count = 0
                        time.sleep(0.01)
                        continue
                    _no_frame_count = 0
                    frame = frame.copy()  # copy agar tidak di-overwrite thread
                else:
                    ret, frame = cap.read()
                    if not ret:
                        if self.source == 'folder':
                            try:
                                cap.release(); cap = self._nf(); continue
                            except StopIteration:
                                print('  Semua video selesai.')
                                break
                        else:
                            break

                self._fn += 1

                # Init ROI sekali saat frame pertama
                if self.roi is None:
                    h,w = frame.shape[:2]
                    self.roi = ROI(w,h)
                    print(f"  ROI: y={self.roi.y}  x={self.roi.x1}..{self.roi.x2}")

                # Deteksi & track
                dets    = self._detect_v(frame) if self._fn % CFG['process_every'] == 0 else []
                tracked = self.tracker.update(dets) if dets else self.tracker.update([])

                # Buat frame ter-anotasi
                annotated = self._annotate(frame.copy(), tracked)

                # Push frame ter-anotasi ke buffer SEBELUM crossing check
                self.buf.push(annotated)

                # Cek crossing MASUK (atas→bawah)
                for det in tracked:
                    x1,y1,x2,y2,cls,conf,tid,prev_y2 = det
                    if (not self.tracker.has_crossed(tid) and
                            self.roi.crossed_downward(prev_y2, y2)):
                        self._on_cross(annotated, frame, det)

                # Feed frame ter-anotasi ke writer aktif
                self.buf.feed(annotated)

                # Tampilkan — resize agar tidak terlalu besar di layar
                disp = annotated
                dh, dw = disp.shape[:2]
                max_w = CFG.get('display_width', 1280)
                if dw > max_w:
                    scale = max_w / dw
                    disp  = cv2.resize(disp, (max_w, int(dh*scale)))
                # Nama window TETAP (tidak berubah tiap frame)
                # sehingga tidak menumpuk saat counter bertambah
                cv2.imshow(self._win_name, disp)
                # Update title bar dengan counter terbaru
                cv2.setWindowTitle(
                    self._win_name,
                    f"ANPR [{CFG['device_id']}]  Masuk: {self.cnt['total']}")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print('\nDihentikan.')
        finally:
            self._rtsp_stop = True   # hentikan thread RTSP reader
            time.sleep(0.3)
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt.pub('status', {
                'device_id': CFG['device_id'],
                'online'   : False,
                'status'   : 0,
            }, retain=True)
            print('Upload terakhir sebelum keluar...')
            try:
                self.uploader.upload_now()   # blocking, tunggu selesai
            except Exception as e:
                print(f'  Upload akhir error: {e}')


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='ANPR Gate Detector - RPi 5')
    ap.add_argument('--config',      default='config.ini',
                    help='Path ke config.ini (default: config.ini di folder script)')
    ap.add_argument('--source',      choices=['rtsp','video','folder','webcam'],
                    default=None)
    ap.add_argument('--file',        type=str,
                    help='Path video file atau folder')
    ap.add_argument('--setup-roi',   action='store_true',
                    help='Mode setup ROI interaktif')
    ap.add_argument('--anpr',        action='store_true',
                    help='Aktifkan ANPR (OCR plat nomor)')
    ap.add_argument('--device',      default=None,
                    help='Override DeviceID')
    ap.add_argument('--vps',         default=None,
                    help='Override RemoteHost dan MQTT Broker')
    ap.add_argument('--upload-every', type=int, default=None,
                    help='Override RsyncInterval (detik)')
    args = ap.parse_args()

    # Load config
    CFG.update(load_config(args.config))

    # Override dari argumen CLI
    if args.source:       CFG['source']       = args.source
    if args.file:         CFG['file_path']    = args.file
    if args.anpr:         CFG['anpr_enabled'] = True
    if args.device:       CFG['device_id']    = args.device
    if args.vps:
        CFG['vps_host']    = args.vps
        CFG['mqtt_broker'] = args.vps
    if args.upload_every: CFG['upload_every_s'] = args.upload_every

    print('=' * 60)
    print(f"ANPR Gate  |  device = {CFG['device_id']}")
    print(f"  VPS      : {CFG['vps_host']}")
    print(f"  ANPR     : {'ON' if CFG['anpr_enabled'] else 'OFF'}")
    print(f"  Upload   : tiap {CFG['upload_every_s']}s")
    print(f"  Lokal    : {CFG['local_clips']}")
    print(f"  Counter  : MASUK SAJA (atas -> bawah ROI)")
    print('=' * 60)

    if args.setup_roi:
        # Tentukan sumber video untuk setup ROI
        if CFG['source'] == 'rtsp' and CFG['rtsp_url']:
            src = CFG['rtsp_url']
            src_label = f"RTSP: {CFG['rtsp_url']}"
        elif CFG['source'] == 'folder' and CFG['file_path']:
            # Ambil video pertama dari folder
            fp = Path(CFG['file_path']) if os.path.isabs(CFG['file_path'])                  else BASE_DIR / CFG['file_path']
            videos = sorted(fp.glob('*.mp4')) + sorted(fp.glob('*.avi'))
            if not videos:
                print(f'Tidak ada video di folder: {fp}')
                raise SystemExit(1)
            src = str(videos[0])
            src_label = f"File: {src}"
        elif CFG['source'] in ('video','webcam') and CFG['file_path']:
            src = CFG['file_path'] if os.path.isabs(CFG['file_path'])                   else str(BASE_DIR / CFG['file_path'])
            src_label = f"File: {src}"
        elif CFG['source'] == 'webcam':
            src = 0
            src_label = "Webcam"
        else:
            print(f"[ERROR] Sumber video tidak diketahui atau RTSP URL kosong.")
            print(f"  Gunakan salah satu:")
            print(f"  python 1_detect.py --setup-roi --source rtsp")
            print(f"  python 1_detect.py --setup-roi --source folder --file videos/")
            print(f"  python 1_detect.py --setup-roi --source video --file test.mp4")
            print(f"  python 1_detect.py --setup-roi --source webcam")
            raise SystemExit(1)

        print(f"  Setup ROI dari: {src_label}")

        # Buka video dan ambil frame
        cap = cv2.VideoCapture(src if isinstance(src, int) else str(src))
        frm = None

        # Untuk RTSP: coba lebih banyak frame karena butuh koneksi
        max_try = 30 if CFG['source'] == 'rtsp' else 10
        for i in range(max_try):
            ret, frm = cap.read()
            if ret:
                break
            time.sleep(0.1)

        cap.release()

        if frm is not None:
            print(f"  Frame berhasil diambil ({frm.shape[1]}x{frm.shape[0]})")
            setup_roi(frm)
        else:
            print(f"[ERROR] Gagal buka sumber video: {src_label}")
            if CFG['source'] == 'rtsp':
                print(f"  Cek: apakah kamera {CFG['rtsp_url']} bisa diakses?")
                print(f"  Test: ffprobe {CFG['rtsp_url']}")
    else:
        Detector(
            source    = CFG['source'],
            file_path = CFG['file_path'],
            anpr      = CFG['anpr_enabled'],
        ).run()