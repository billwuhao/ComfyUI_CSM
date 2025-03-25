import numpy as np
import torch
import time
import librosa
import sounddevice as sd
from scipy import ndimage
from comfy.utils import ProgressBar


class AudioRecorderCSM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Trigger control
                "trigger": ("BOOLEAN", {"default": False}),
                # Recording duration
                "record_sec": ("INT", {
                    "default": 5,
                    "min": 1,
                    "step": 1  # integer seconds increment
                }),
                # Standard sample rates selection
                "sample_rate": (["16000", "44100", "48000"], {
                    "default": "48000"
                }),
                # FFT size (must be power of 2)
                "n_fft": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 512  # 512, 1024, 1536...4096
                }),
                # Noise gate sensitivity
                "sensitivity": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1  # 0.1 increments
                }),
                # Smoothing kernel size (must be odd)
                "smooth": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 7,
                    "step": 2  # generates 1,3,5,7
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "interlocutor": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio", "interlocutor")
    FUNCTION = "record_and_clean"
    CATEGORY = "🎤MW/MW-CSM"

    def _stft(self, y, n_fft):
        """Compute STFT with 25% overlap"""
        hop = n_fft // 4
        return librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft)

    def _istft(self, spec, n_fft):
        """Inverse STFT with 25% overlap"""
        hop = n_fft // 4
        return librosa.istft(spec, hop_length=hop, win_length=n_fft)

    def _calc_noise_profile(self, noise_clip, n_fft):
        """Calculate noise profile statistics from reference clip"""
        noise_spec = self._stft(noise_clip, n_fft)
        return {
            'mean': np.mean(np.abs(noise_spec), axis=1, keepdims=True),
            'std': np.std(np.abs(noise_spec), axis=1, keepdims=True)
        }

    def _spectral_gate(self, spec, noise_profile, sensitivity):
        """Apply spectral gating with dynamic threshold"""
        threshold = noise_profile['mean'] + sensitivity * noise_profile['std']
        return np.where(np.abs(spec) > threshold, spec, 0)

    def _smooth_mask(self, mask, kernel_size):
        """Apply smoothing filter to binary mask"""
        smoothed = ndimage.uniform_filter(mask, size=(kernel_size, kernel_size))
        return np.clip(smoothed * 1.2, 0, 1)  # enhance edge preservation

    def record_and_clean(self, trigger, record_sec, n_fft, sensitivity, smooth, sample_rate, interlocutor=None, seed=0):
        if not trigger:
            if interlocutor is not None:
                return (None, interlocutor)
            raise ValueError("No trigger received: Recording not opened.")

        sr = int(sample_rate)
        final_audio = None

        try:
            noise_clip = None
            # Main recording process
            main_rec = sd.rec(int(record_sec * sr), samplerate=sr, channels=1, dtype='float32')
            pb = ProgressBar(record_sec)
            for _ in range(record_sec * 2):
                time.sleep(0.5)
                pb.update(0.5)
            sd.wait()
            audio = main_rec.flatten()

            # Automatic noise detection
            if noise_clip is None:
                energy = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=n_fft//4)
                min_idx = np.argmin(energy)
                start = min_idx * (n_fft//4)
                noise_clip = audio[start:start + n_fft*2]

            # Noise reduction pipeline
            noise_profile = self._calc_noise_profile(noise_clip, n_fft)
            spec = self._stft(audio, n_fft)
            
            # Multi-stage processing
            mask = np.ones_like(spec)  # Initial mask
            for _ in range(2):  # Dual processing loop
                cleaned_spec = self._spectral_gate(spec, noise_profile, sensitivity)
                mask = np.where(np.abs(cleaned_spec) > 0, 1, 0)
                mask = self._smooth_mask(mask, smooth//2+1)
                spec = spec * mask

            # Phase reconstruction
            processed = self._istft(spec * mask, n_fft)
            
            # Dynamic gain normalization
            peak = np.max(np.abs(processed))
            processed = processed * (0.99 / peak) if peak > 0 else processed

            # Format conversion
            waveform = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
            final_audio = {"waveform": waveform, "sample_rate": sr}

        except Exception as e:
            print(f"Recording/processing failed: {str(e)}")
            raise

        return (final_audio, interlocutor)