
import numpy as np

PAPER_FPS = 10              
PAPER_SEQ_LEN = 300         
RPPG_DIM = 64               
HRV_BANDS = {               
    'vlf': (0.003, 0.04),
    'lf': (0.04, 0.15),
    'hf': (0.15, 0.4),
}


def extract_rppg_signal(frames):
       if isinstance(frames, np.ndarray) and frames.ndim == 4:
        green = frames[:, :, :, 1]
        signal = green.mean(axis=(1, 2))
    else:
        signal = np.zeros(PAPER_SEQ_LEN, dtype=np.float32)
    signal = signal.astype(np.float32)
    signal -= signal.mean()
    std = signal.std()
    if std > 1e-8:
        signal /= std
    return signal


def compute_hrv_features(rppg_signal, fps=PAPER_FPS):
        signal = np.asarray(rppg_signal, dtype=np.float32)
    peaks = _find_peaks_simple(signal)
    if len(peaks) < 2:
        return _fallback_features(signal)

    rr = np.diff(peaks) / fps
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 2:
        return _fallback_features(signal)

    mean_rr = float(np.mean(rr))
    std_rr = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    pnn50 = float(np.sum(np.abs(np.diff(rr)) > 0.05) / max(len(rr) - 1, 1))
    hr = 60.0 / (mean_rr + 1e-8)

    time_feats = [mean_rr, std_rr, rmssd, pnn50, hr]
    freq_feats = _freq_domain_features(signal, fps)
    stat_feats = [
        float(signal.mean()), float(signal.std()),
        float(np.percentile(signal, 25)), float(np.percentile(signal, 75)),
        float(signal.max()), float(signal.min()),
    ]

    raw = np.array(time_feats + freq_feats + stat_feats, dtype=np.float32)
    feat = np.zeros(RPPG_DIM, dtype=np.float32)
    feat[: min(len(raw), RPPG_DIM)] = raw[:RPPG_DIM]
    return feat


def _find_peaks_simple(signal, min_distance=5):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def _freq_domain_features(signal, fps):
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    vlf = fft_vals[(freqs >= HRV_BANDS['vlf'][0]) & (freqs < HRV_BANDS['vlf'][1])].sum()
    lf = fft_vals[(freqs >= HRV_BANDS['lf'][0]) & (freqs < HRV_BANDS['lf'][1])].sum()
    hf = fft_vals[(freqs >= HRV_BANDS['hf'][0]) & (freqs < HRV_BANDS['hf'][1])].sum()
    total = vlf + lf + hf + 1e-8
    return [vlf / total, lf / total, hf / total, lf / (hf + 1e-8)]


def _fallback_features(signal):
    feat = np.zeros(RPPG_DIM, dtype=np.float32)
    feat[0] = float(signal.mean()) if len(signal) else 0.0
    feat[1] = float(signal.std()) if len(signal) else 0.0
    feat[2] = float(len(signal))
    return feat


def extract_hrv_from_tensor(frames_tensor, fps=PAPER_FPS):
    frames = frames_tensor.permute(1, 2, 3, 0).cpu().numpy()
    rppg = extract_rppg_signal(frames)
    return compute_hrv_features(rppg, fps)


def get_paper_hrv_template(fatigue_stage='Alert'):
    feat = np.zeros(RPPG_DIM, dtype=np.float32)
    if fatigue_stage == 'Alert':
        feat[0], feat[4], feat[8], feat[9], feat[10] = 0.85, 65.0, 0.15, 0.35, 0.45
    elif fatigue_stage == 'Drowsy':
        feat[0], feat[4], feat[8], feat[9], feat[10] = 0.95, 58.0, 0.20, 0.45, 0.35
    else:
        feat[0], feat[4], feat[8], feat[9], feat[10] = 1.05, 52.0, 0.28, 0.55, 0.25
    return feat
