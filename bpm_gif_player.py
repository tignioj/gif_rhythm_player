import cv2
import time
import numpy as np
import os
import threading
from collections import deque
import sounddevice as sd
import essentia.standard as es

# ---------- 全局共享变量 ----------
current_bpm = 120.0          # 默认 BPM
bpm_lock = threading.Lock()  # 保护 current_bpm
running = True              # 控制线程退出

# ---------- BPM 检测参数 ----------
SAMPLE_RATE = 22050
# SAMPLE_RATE = 48000
WINDOW_SECONDS = 8
BLOCK_SIZE = 1024
CHANNELS = 1

audio_buffer = deque()
buffer_lock = threading.Lock()

# ---------- 音频回调 ----------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️ 状态：{status}")
    with buffer_lock:
        audio = indata.copy().astype(np.float32)
        if audio.shape[1] > 1:
            audio = np.mean(audio, axis=1, keepdims=True)
        audio_buffer.append((time.time(), audio.flatten()))

# ---------- Essentia BPM 估算 ----------
def estimate_bpm(audio_signal, sample_rate=SAMPLE_RATE):
    bpm_estimator = es.PercivalBpmEstimator(sampleRate=sample_rate)
    try:
        return bpm_estimator(audio_signal)
    except Exception as e:
        print(f"❌ BPM 估算失败：{e}")
        return 0.0

# ---------- 获取最近窗口音频 ----------
def get_recent_audio():
    with buffer_lock:
        if not audio_buffer:
            return None
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        chunks = [block for ts, block in audio_buffer if ts >= cutoff]
        if not chunks:
            return None
        full_audio = np.concatenate(chunks)
        return full_audio

# ---------- BPM 检测线程函数 ----------
def bpm_detection_loop():
    global current_bpm
    print("🎤 启动 BPM 检测线程...")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        callback=audio_callback,
        dtype='float32'
    )
    with stream:
        while running:
            time.sleep(WINDOW_SECONDS)
            audio = get_recent_audio()
            if audio is None or len(audio) < SAMPLE_RATE * 0.5:
                print("⏳ 等待足够音频数据...")
                continue
            audio = np.clip(audio, -1.0, 1.0)
            bpm = estimate_bpm(audio, SAMPLE_RATE)
            if bpm > 0:
                with bpm_lock:
                    current_bpm = bpm
                print(f"🎵 更新 BPM 为：{bpm:.1f}")
            else:
                print("⚠️ 无法检测稳定节拍")

# ---------- 播放 GIF 函数（实时读取全局 BPM）----------
def play_gif_with_beat_pattern(
    gif_path,
    beat_pattern=(1, 0, 1, 0),
    frames_per_beat=6,
):
    global running
    cap = cv2.VideoCapture(gif_path)
    if not cap.isOpened():
        print("❌ 无法打开 GIF")
        return

    gif_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if gif_frame_count <= 0:
        print("❌ 无效的 GIF 帧数")
        return

    pattern_len = len(beat_pattern)
    start_time = time.perf_counter()

    print(f"▶️ 开始播放 GIF，初始 BPM: {current_bpm:.1f}")
    print(f"拍子模式: {beat_pattern}")
    print(f"每拍帧数: {frames_per_beat}")

    while running:
        # 1. 获取当前最新的 BPM
        with bpm_lock:
            bpm = current_bpm

        beat_interval = 60.0 / bpm

        # 2. 时间计算
        now = time.perf_counter()
        elapsed = now - start_time

        beat_index_global = int(elapsed / beat_interval)
        beat_index = beat_index_global % pattern_len
        is_strong = beat_pattern[beat_index] == 1

        beat_phase = (elapsed % beat_interval) / beat_interval
        frame_in_beat = int(beat_phase * frames_per_beat)
        frame_in_beat = min(frame_in_beat, frames_per_beat - 1)

        gif_frame_index = (
            beat_index_global * frames_per_beat + frame_in_beat
        ) % gif_frame_count

        # 3. 读取并显示帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, gif_frame_index)
        # ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (240,240))

        # 显示当前 BPM 和拍信息
        # cv2.putText(frame, f"BPM: {bpm:.1f}", (40, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # cv2.putText(frame, f"Beat frame: {frame_in_beat+1}/{frames_per_beat}",
        #             (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # cv2.putText(frame, f"Pattern idx: {beat_index}",
        #             (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow("GIF Beat Sync (Live BPM)", frame)

        # 4. 按 ESC 退出
        if cv2.waitKey(1) == 27:
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- 主程序 ----------
if __name__ == "__main__":
    # gif_path = "resources/gif/dance4.gif"
    gif_path = "resources/gif/common.gif"
    # gif_path = os.path.join(application_path, "resources/gif/common.gif")

    # 启动 BPM 检测线程（后台）
    bpm_thread = threading.Thread(target=bpm_detection_loop, daemon=True)
    bpm_thread.start()

    # 主线程播放 GIF（阻塞）
    play_gif_with_beat_pattern(
        gif_path,
        beat_pattern=(1, 0, 1, 0),
        frames_per_beat=13,
    )