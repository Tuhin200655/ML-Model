import librosa
import numpy as np

def extract_audio_stress_features(audio_path):
    """
    Analyzes an audio file to extract indicators of stress, trauma, and anxiety.
    Focuses on pitch variation, pauses, and energy.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        if len(y) == 0:
            return None

        # 1. Pitch Analysis (F0)
        # High pitch or extreme variance often indicates acute stress/anxiety
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Get the average pitch of the strongest components
        avg_pitch = np.mean(pitches[magnitudes > np.max(magnitudes) * 0.1]) if np.any(magnitudes > 0) else 0

        # 2. Pause Analysis (Silence Detection)
        # Long or erratic pauses can indicate severe trauma or hesitation
        # intervals returns non-silent intervals
        intervals = librosa.effects.split(y, top_db=30)
        total_duration = len(y) / sr
        silent_duration = total_duration - sum([i[1] - i[0] for i in intervals]) / sr
        pause_ratio = silent_duration / total_duration if total_duration > 0 else 0

        # 3. Energy/Intensity (RMS)
        # Erratic volume or very low energy can indicate depression/withdrawal
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)

        return {
            "avg_pitch": float(avg_pitch),
            "pause_ratio": float(pause_ratio),
            "avg_energy": float(avg_energy),
            "status": "success"
        }

    except Exception as e:
        print(f"Audio processing error: {e}")
        return None

def calculate_audio_stress_score(features):
    """
    Normalizes audio features into a stress bonus (0.0 to 1.0).
    """
    if not features:
        return 0.0

    score = 0.0

    # High pause ratio (e.g., > 30%) indicates hesitation/trauma
    if features['pause_ratio'] > 0.3:
        score += 0.4
    elif features['pause_ratio'] > 0.15:
        score += 0.2

    # High pitch deviation (simplified: high avg pitch for human speech)
    # Normal human speech is usually 100-300Hz. Extreme spikes indicate stress.
    if features['avg_pitch'] > 300:
        score += 0.3
    elif features['avg_pitch'] < 80: # Very low, flat tone (depression)
        score += 0.2

    # Low energy (withdrawal/depression)
    if features['avg_energy'] < 0.02:
        score += 0.3

    return min(1.0, score)
