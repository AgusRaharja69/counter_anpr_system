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
    Rekam satu segment via ffmpeg menggunakan dua tahap:
    1. Rekam ke format .ts (MPEG-TS) — tidak butuh rewrite header, selalu berhasil
    2. Remux .ts ke .mp4 tanpa re-encode — cepat, < 1 detik
    Strategi ini menghindari error "Unable to re-open" pada mp4 langsung.
    """
    Path(videos_dir).mkdir(parents=True, exist_ok=True)
    ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
    ts_file  = str(Path(videos_dir) / f'tmp_{ts_str}.ts')   # intermediate
    mp4_file = str(Path(videos_dir) / f'rec_{ts_str}.mp4')  # output akhir

    # ── Tahap 1: Rekam ke .ts (stream copy, tidak ada rewrite header) ──
    cmd_rec = [
        'ffmpeg',
        '-loglevel',       'warning',      # tampilkan warning untuk debug
        '-rtsp_transport', 'tcp',
        '-fflags',         'nobuffer',
        '-flags',          'low_delay',
        '-i',              rtsp_url,
        '-t',              str(duration),
        '-c:v',            'copy',
        '-an',
        '-f',              'mpegts',       # format TS tidak butuh rewrite
        '-y',
        ts_file,
    ]

    log_file = str(Path(videos_dir) / f'rec_{ts_str}.log')
    print(f'  [rec] START rec_{ts_str}.mp4  ({duration}s)')
    try:
        # Tulis stderr ke file log agar tidak buffer-blocking
        with open(log_file, 'w') as lf:
            r1 = subprocess.run(cmd_rec,
                                stdout=subprocess.DEVNULL,
                                stderr=lf,
                                timeout=duration + 30)

        # Baca log untuk cek error
        log_content = ''
        try:
            log_content = open(log_file).read()
        except: pass

        # Cek hasil rekam
        if not Path(ts_file).exists() or Path(ts_file).stat().st_size < 1024:
            lines = [l for l in log_content.strip().splitlines() if l.strip()]
            for l in lines[-4:]:
                print(f'  [rec] ERR: {l}')
            try: Path(log_file).unlink()
            except: pass
            return None

        size_ts = Path(ts_file).stat().st_size / 1024 / 1024

        # ── Tahap 2: Remux .ts → .mp4 (sangat cepat, tidak re-encode) ──
        cmd_mux = [
            'ffmpeg',
            '-loglevel', 'error',
            '-i',        ts_file,
            '-c',        'copy',
            '-y',
            mp4_file,
        ]
        r2 = subprocess.run(cmd_mux,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=30)

        # Hapus file .ts sementara
        try: Path(ts_file).unlink()
        except: pass

        # Hapus log setelah selesai
        try: Path(log_file).unlink()
        except: pass

        if r2.returncode == 0 and Path(mp4_file).exists():
            size_mp4 = Path(mp4_file).stat().st_size / 1024 / 1024
            print(f'  [rec] DONE  rec_{ts_str}.mp4  ({size_mp4:.1f} MB)')
            return mp4_file
        else:
            err = r2.stderr.decode(errors='replace').strip() if r2.stderr else ''
            print(f'  [rec] REMUX FAIL: {err[:150]}')
            # Fallback: rename .ts → .mp4 agar 1_detect.py tetap bisa baca
            if Path(ts_file).exists():
                Path(ts_file).rename(mp4_file)
            return mp4_file if Path(mp4_file).exists() else None

    except subprocess.TimeoutExpired:
        print(f'  [rec] TIMEOUT rec_{ts_str}.mp4')
        for f in [ts_file, mp4_file]:
            try: Path(f).unlink()
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