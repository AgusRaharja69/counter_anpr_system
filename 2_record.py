"""
record.py  —  Raspberry Pi 5
==============================
Rekam RTSP stream terus-menerus, potong tiap N detik,
simpan ke folder videos/ untuk diproses 1_detect.py.

Jalankan DUA terminal:
  Terminal 1: python record.py
  Terminal 2: python 1_detect.py --source folder --file videos/

Alur:
  RTSP → ffmpeg → videos/rec_YYYYMMDD_HHMMSS.mp4  (tiap 60 detik)
  1_detect.py baca folder videos/ → proses → HAPUS file setelah selesai

Config dibaca dari config.ini section [CAMERA] dan [RECORD].
"""

import os, sys, time, subprocess, argparse, threading, shutil
from datetime import datetime
from pathlib import Path
from configparser import ConfigParser

BASE_DIR = Path(__file__).parent.resolve()

# ─── LOAD CONFIG ──────────────────────────────────────────────────────────────
def load_config(path='config.ini'):
    p = Path(path) if Path(path).is_absolute() else BASE_DIR / path
    ini = ConfigParser()
    if not p.exists():
        print(f'[ERROR] config.ini tidak ditemukan: {p}')
        raise SystemExit(1)
    ini.read(str(p), encoding='utf-8')

    def get(s, k, fb=''):
        return ini.get(s, k, fallback=fb).strip()
    def getint(s, k, fb):
        try:    return int(get(s, k, str(fb)))
        except: return fb

    raw_videos = get('RECORD', 'VideoPath', fb='videos')
    videos_dir = raw_videos if os.path.isabs(raw_videos) \
                 else str(BASE_DIR / raw_videos)

    return {
        'rtsp_url'    : get('CAMERA', 'RTSP',
                            fb='rtsp://admin:123456@192.168.0.224:554/'
                               'Streaming/Channels/101'),
        'videos_dir'  : videos_dir,
        'duration'    : getint('RECORD', 'Duration',      fb=60),
        'keep_files'  : getint('RECORD', 'KeepFiles',     fb=10),
        'reconnect_s' : getint('CAMERA', 'ReconnectDelay', fb=5),
        'record_fps'  : getint('RECORD', 'RecordFPS',     fb=10),
    }


CFG   = {}
_stop = False

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
def cleanup_old(videos_dir, keep):
    """Hapus file lama jika melebihi batas keep_files."""
    files = sorted(
        Path(videos_dir).glob('rec_*.mp4'),
        key=lambda f: f.stat().st_mtime)
    to_del = files[:-keep] if keep > 0 else []
    for f in to_del:
        try:
            f.unlink()
            print(f'  [cleanup] hapus lama: {f.name}')
        except Exception as e:
            print(f'  [cleanup] gagal hapus {f.name}: {e}')


# ─── RECORD ONE SEGMENT ───────────────────────────────────────────────────────
def record_segment(videos_dir, duration, rtsp_url):
    """
    Rekam satu segment via ffmpeg.
    Return: path file yang baru selesai, atau None jika gagal.
    """
    Path(videos_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rec_{ts}.mp4'
    filepath = str(Path(videos_dir) / filename)

    fps = CFG.get('record_fps', 10)

    # Jika fps < fps kamera asli: re-encode dengan fps rendah
    # Manfaat: file lebih kecil + 1_detect.py proses lebih sedikit frame
    # tapi setiap frame tetap resolusi penuh (tidak ada informasi hilang)
    if fps > 0:
        video_opts = [
            '-vf',   f'fps={fps}',       # throttle ke N fps
            '-c:v',  'libx264',          # re-encode (perlu karena ubah fps)
            '-preset', 'ultrafast',      # encode secepat mungkin
            '-crf',  '28',               # kualitas: 18=tinggi, 28=cukup
        ]
    else:
        # fps=0: copy stream langsung tanpa re-encode (paling cepat)
        video_opts = ['-c:v', 'copy']

    cmd = [
        'ffmpeg',
        '-loglevel',       'error',
        '-rtsp_transport', 'tcp',
        '-fflags',         'nobuffer',
        '-flags',          'low_delay',
        '-i',              rtsp_url,
        '-t',              str(duration),
        *video_opts,
        '-an',
        '-movflags', '+faststart',
        '-y',
        filepath,
    ]

    print(f'  [rec] START {filename}  ({duration}s)')
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=duration + 30)

        if result.returncode == 0 and Path(filepath).exists():
            size_mb = Path(filepath).stat().st_size / 1024 / 1024
            print(f'  [rec] DONE  {filename}  ({size_mb:.1f} MB)')
            return filepath
        else:
            err = result.stderr.decode(errors='replace')[:200]
            print(f'  [rec] FAIL  {filename}  → {err}')
            # Hapus file rusak jika ada
            try: Path(filepath).unlink()
            except: pass
            return None

    except subprocess.TimeoutExpired:
        print(f'  [rec] TIMEOUT {filename}')
        try: Path(filepath).unlink()
        except: pass
        return None
    except FileNotFoundError:
        print('[ERROR] ffmpeg tidak ditemukan. Install: sudo apt install ffmpeg')
        raise SystemExit(1)
    except Exception as e:
        print(f'  [rec] ERROR: {e}')
        return None


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    global _stop

    print('=' * 50)
    print('RTSP Recorder')
    print(f"  RTSP     : {CFG['rtsp_url']}")
    print(f"  Simpan ke: {CFG['videos_dir']}")
    print(f"  Durasi   : {CFG['duration']}s per segment")
    print(f"  FPS      : {CFG['record_fps']} fps (0=copy asli)")
    print(f"  Keep     : {CFG['keep_files']} file terakhir")
    print('=' * 50)
    print('Ctrl+C untuk berhenti\n')

    fail_count = 0

    while not _stop:
        path = record_segment(
            CFG['videos_dir'],
            CFG['duration'],
            CFG['rtsp_url'])

        if path:
            fail_count = 0
            # Bersihkan file lama
            cleanup_old(CFG['videos_dir'], CFG['keep_files'])
        else:
            fail_count += 1
            wait = min(CFG['reconnect_s'] * fail_count, 30)
            print(f'  [rec] Gagal {fail_count}x, tunggu {wait}s...')
            time.sleep(wait)

    print('\nRecorder berhenti.')


# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RTSP Recorder for RPi 5')
    ap.add_argument('--config',   default='config.ini')
    ap.add_argument('--duration', type=int, default=None,
                    help='Override durasi segment (detik)')
    ap.add_argument('--keep',     type=int, default=None,
                    help='Override jumlah file yang disimpan')
    ap.add_argument('--rtsp',     default=None,
                    help='Override RTSP URL')
    args = ap.parse_args()

    CFG.update(load_config(args.config))
    if args.duration: CFG['duration']   = args.duration
    if args.keep:     CFG['keep_files'] = args.keep
    if args.rtsp:     CFG['rtsp_url']   = args.rtsp

    try:
        main()
    except KeyboardInterrupt:
        _stop = True
        print('\nDihentikan.')