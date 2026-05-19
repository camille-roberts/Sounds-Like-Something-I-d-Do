import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import librosa, librosa.display
import soundfile as sf


VIDEO_PATH = "so_sorry.mov"

# Load video and extract audio
clip = VideoFileClip(VIDEO_PATH)
audio = clip.audio
audio.write_audiofile("audio_full.wav", fps=16000, codec="pcm_s16le")

# Load audio                                     n
y_full, sr = librosa.load("audio_full.wav", sr=16000)

# Create time axis for full waveform
time_full = np.arange(len(y_full)) / sr

# --------------------------------------------------------
# FIND "SO SORRY!" MOMENT
# (We manually set a time window; adjust if needed.)
# --------------------------------------------------------
start_sec = 8.8   # start of "so sorry"
end_sec   = 10   # "end of so sorry"

start = int(start_sec * sr)
end = int(end_sec * sr)

y_zoom = y_full[start:end]
t_zoom = time_full[start:end]

# FEATURE EXTRACTIONS


# Amplitude envelope (loudness)
frame = 2048
hop = 512
amp_env = librosa.feature.rms(y=y_full, frame_length=frame, hop_length=hop)[0]
times_env = librosa.frames_to_time(np.arange(len(amp_env)), sr=sr, hop_length=hop)

# Pitch (fundamental frequency)
f0, voiced_flag, voiced_probs = librosa.pyin(
    y_full, fmin=80, fmax=400, frame_length=frame, hop_length=hop
)
times_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)


plt.figure(figsize=(12, 3))
plt.plot(time_full, y_full)
plt.title("Full Waveform (Time-Aligned)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim([0, time_full[-1]])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(times_env, amp_env)
plt.title("Amplitude Envelope (Loudness)")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Energy")
plt.xlim([0, times_env[-1]])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(times_f0, f0)
plt.title("Pitch (F0 Contour)")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.xlim([0, times_f0[-1]])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(t_zoom, y_zoom)
plt.title("Waveform During 'So Sorry!'")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.xlim([t_zoom[0], t_zoom[-1]])
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# SAVE CLIP AUDIO SNIPPET
# --------------------------------------------------------
sf.write("so_sorry_clip.wav", y_zoom, sr)
plt.savefig("waveform_full.png", dpi=300)
plt.savefig("waveform_zoom_sosorry.png", dpi=300)
plt.savefig("pitch_contour.png", dpi=300)
plt.savefig("loudness_rms.png", dpi=300)
print("✅ Saved emotion clip: so_sorry_clip.wav")
