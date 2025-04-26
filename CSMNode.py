from dataclasses import dataclass
from typing import List, Tuple
import ast
import silentcipher
import torch
import torchaudio
import os

from huggingface_hub import hf_hub_download
from .models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import folder_paths


models_dir = folder_paths.models_dir

class AddWatermark:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        self.cached_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "add_watermark": ("BOOLEAN", {
                        "default": False, 
                        "tooltip": "Enable audio watermark embedding"
                    }),
                    "key": ("STRING", {
                        "default": "[212, 211, 146, 56, 201]", 
                        "tooltip": "Encryption key as list of integers (e.g. [212,211,146,56,201])"
                    }),
                    "unload_model": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Unload model from memory after use"
                    })
                    },
                "optional": {}
                }


    CATEGORY = "🎤MW/MW-CSM"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "watermark")
    FUNCTION = "watermarkgen"

    def watermarkgen(self, audio, add_watermark, key, unload_model):
        """Main watermark processing pipeline"""
        ckpt_path = os.path.join(models_dir, "TTS", "SilentCipher", "44_1_khz", "73999_iteration")
        config_path = os.path.join(models_dir, ckpt_path, "hparams.yaml")

        if self.cached_model is None:
            self.cached_model = silentcipher.get_model(
                model_type="44.1k", 
                ckpt_path=ckpt_path, 
                config_path=config_path,
                device=self.device,
            )
        
        watermarker = self.cached_model

        audio_array, sample_rate = self.load_audio(audio)
        # Ensure tensor on correct device
        audio_array = audio_array.to(self.device)

        if add_watermark:
            key = self._parse_key(key)
            audio_array, sample_rate = self.watermark(watermarker, audio_array, sample_rate, key)

        watermark = self.verify(watermarker, audio_array, sample_rate)
        
        if unload_model:
            del watermarker
            self.cached_model = None
            torch.cuda.empty_cache()
        
        # Move data back to CPU before return
        return ({"waveform": audio_array.unsqueeze(0).unsqueeze(0).cpu(), "sample_rate": sample_rate}, watermark)

    @torch.inference_mode()
    def watermark(self,
        watermarker: silentcipher.server.Model,
        audio_array: torch.Tensor,
        sample_rate: int,
        watermark_key: list[int],
    ) -> tuple[torch.Tensor, int]:
        # Ensure mono channel
        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_array = audio_array.mean(dim=0)
        
        audio_array = audio_array.to(self.device)
        
        audio_array_44khz = torchaudio.functional.resample(
            audio_array, 
            orig_freq=sample_rate, 
            new_freq=44100
        ).to(self.device)
        
        # Ensure correct tensor shape (should be 1D)
        if len(audio_array_44khz.shape) != 1:
            audio_array_44khz = audio_array_44khz.reshape(-1)
            
        try:
            # Enhance watermark strength by reducing SDR threshold
            encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=30)
            
            verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)
            
            if not verify_result["status"]:
                encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=25)
                verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)
        except Exception as e:
            return audio_array, sample_rate

        # Resample back to original rate if needed
        output_sample_rate = min(44100, sample_rate)
        if output_sample_rate != 44100:
            encoded = torchaudio.functional.resample(
                encoded, 
                orig_freq=44100, 
                new_freq=output_sample_rate
            ).to(self.device)

        return encoded, output_sample_rate

    @torch.inference_mode()
    def verify(self,
        watermarker: silentcipher.server.Model,
        watermarked_audio: torch.Tensor,
        sample_rate: int,
    ) -> str:
        if len(watermarked_audio.shape) > 1 and watermarked_audio.shape[0] > 1:
            watermarked_audio = watermarked_audio.mean(dim=0)
            
        if sample_rate != 44100:
            watermarked_audio_44khz = torchaudio.functional.resample(
                watermarked_audio, 
                orig_freq=sample_rate, 
                new_freq=44100
            ).to(self.device)
        else:
            watermarked_audio_44khz = watermarked_audio.to(self.device)
        
        if len(watermarked_audio_44khz.shape) != 1:
            watermarked_audio_44khz = watermarked_audio_44khz.reshape(-1)
            
        
        # 尝试不同的解码参数
        # 1. 使用相位偏移解码
        result_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)
        
        # 2. 不使用相位偏移解码
        result_no_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=False)
        
        # 使用两种方法中任一种成功的结果
        if result_phase["status"]:
            watermark = "Watermarked:" + str(result_phase["messages"][0])
        elif result_no_phase["status"]:
            watermark = "Watermarked:" + str(result_no_phase["messages"][0])
        else:
            watermark = "No watermarked"

        return watermark

    def _parse_key(self, key_string):
        """Safely parse encryption key from string
        Args:
            key_string: String representation of key list
        Returns:
            List[int]: Parsed key sequence
        """
        try:
            return ast.literal_eval(key_string)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid key format: {str(e)}")


    def load_audio(self, audio) -> tuple[torch.Tensor, int]:
        waveform = audio["waveform"].squeeze(0)
        audio_array = waveform.mean(dim=0)
        sample_rate = audio["sample_rate"]
        return audio_array, int(sample_rate)


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor

SEGMENTS = []
SPEAKERS = []
    
class Generator:
    def __init__(
        self,
        model: Model,
        device: str = "cuda",
    ):
        self._model = model
        self._model.setup_caches(1)
        self.device = device

        self._text_tokenizer = self.load_llama3_tokenizer()

        mimi = self.load_mimi()
        self._audio_tokenizer = mimi
        self.sample_rate = mimi.sample_rate

    def clean_memory(self):
        self._model = None
        self._text_tokenizer = None
        self._audio_tokenizer = None
        self.sample_rate = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def load_llama3_tokenizer(self):
        """
        https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
        """
        llama_path = os.path.join(models_dir, "LLM", "Llama-3.2-1B")
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )

        return tokenizer

    def load_mimi(self):
        mimi_path = os.path.join(models_dir, "TTS", "moshiko-pytorch-bf16", loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_path, device=self.device)
        mimi.set_num_codebooks(32)

        return mimi

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        return audio


class MultiLinePromptCSM:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-CSM"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


class CSMDialogRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.csm_1b_cached_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING",{"forceInput": True}),
                    "unload_speakers": ("BOOLEAN",{ "default": False}),
                    "unload_model": ("BOOLEAN", {
                            "default": True,
                            "tooltip": "Unload model from memory after use"
                    }),
                    },
                "optional": {
                    "prompt0": ("STRING",),
                    "prompt1": ("STRING",),
                    "prompt2": ("STRING",),
                    "prompt3": ("STRING",),
                    "audio0": ("AUDIO",),
                    "audio1": ("AUDIO",),
                    "audio2": ("AUDIO",),
                    "audio3": ("AUDIO",),
                    "who_will_speak": ("INT", {
                        "default": 0, 
                        "min": 0, 
                        "max": 9, 
                        "step": 1
                    }),
                    "max_audio_length_ms": ("INT", {
                        "default": 1000,
                        "min": 500,
                        "max": 120_000,
                        "step": 500
                    }),
                    "temperature": ("FLOAT", {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01
                    }),
                    "topk": ("INT", {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1
                    }),
                }
        }


    CATEGORY = "🎤MW/MW-CSM"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "prompt")
    FUNCTION = "run"


    def run(self, 
            text, 
            unload_speakers, 
            unload_model, 
            prompt0="", 
            prompt1="", 
            prompt2="", 
            prompt3="", 
            audio0=None, 
            audio1=None, 
            audio2=None, 
            audio3=None, 
            who_will_speak=1,
            max_audio_length_ms=90_000,
            temperature=0.9,
            topk=50,
            ):
        """Main dialog generation pipeline
        Args:
            text: Input text to be synthesized
            unload_speakers: Flag to clear speaker history
            prompt0-3: Context prompts for dialogue generation
            audio0-3: Reference audio clips for speaker style
            who_will_speak: Selected speaker ID for synthesis
        """
        generator = Generator(self.load_csm_1b(), device=self.device)
        global SEGMENTS, SPEAKERS
        if unload_speakers:
            SEGMENTS.clear()
            SPEAKERS.clear()
            
        # Process context inputs
        segments = []
        for i in range(4):
            prompt = locals()[f"prompt{i}"]
            audio = locals()[f"audio{i}"]
            # print(f"prompt{i}: {prompt}→audio{i}")
            if audio is not None:
                audio_tensor = audio["waveform"].squeeze(0).mean(dim=0)
                sample_rate = int(audio["sample_rate"])
                # print(f"audio{i} sample_rate: {sample_rate}")
                audio_tensor = torchaudio.functional.resample(
                                                            audio_tensor.squeeze(0), 
                                                            orig_freq=sample_rate, 
                                                            new_freq=generator.sample_rate
                                                        )
            else:
                audio_tensor = None
            
            speaker, prompt = self.get_speaker_text(prompt)

            segment = self.get_segment(speaker, prompt, audio_tensor)
            if segment is not None:
                SEGMENTS.append(segment)
                SPEAKERS.append(speaker)
        
        if SEGMENTS:
            # Generate with context
            audio = generator.generate(
                text=text,
                speaker=who_will_speak,
                context=SEGMENTS,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk,
            )
            out_prompt = f"{who_will_speak}: {text}"
        else:
            # Generate without context
            audio = generator.generate(
                text=text,
                speaker=0,
                context=[],
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                topk=topk,
            )
            out_prompt = f"0: {text}"

        sr = generator.sample_rate

        if unload_model:
            generator.clean_memory()
            del generator
            self.csm_1b_cached_model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        return ({"waveform": audio.unsqueeze(0).unsqueeze(0).cpu(), "sample_rate": sr}, out_prompt)

    def get_speaker_text(self, text):
        import re
        if text.strip() != "":
            st = [i.strip() for i in re.split('[:：]', text, 1)]
            if len(st) == 2:
                speaker, prompt = st
                return int(speaker), prompt
            else:
                raise ValueError("Invalid text format")
        else:
            return None, None
    
    def get_segment(self, speaker, text, audio):
        if speaker is not None:
            if audio is not None:
                return Segment(speaker=speaker, text=text, audio=audio)
            else:
                raise ValueError(f"{text}: Audio is required")
        else:
            return None

    def load_csm_1b(self):
        if self.csm_1b_cached_model is not None:
            return self.csm_1b_cached_model
        else:
            csm_1b_path = os.path.join(models_dir, "TTS", "csm-1b")
            config_path = os.path.join(csm_1b_path, "config.json")
            import json
            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            config = config["args"]
            configs = ModelArgs(backbone_flavor = config["backbone_flavor"], 
                                decoder_flavor = config["decoder_flavor"], 
                                text_vocab_size = config["text_vocab_size"], 
                                audio_vocab_size = config["audio_vocab_size"], 
                                audio_num_codebooks = config["audio_num_codebooks"])
            self.csm_1b_cached_model = Model.from_pretrained(csm_1b_path, config=configs)
            self.csm_1b_cached_model.to(device=self.device, dtype=torch.bfloat16)

            return self.csm_1b_cached_model

from .MWAudioRecorderCSM import AudioRecorderCSM

NODE_CLASS_MAPPINGS = {
    "AddWatermark": AddWatermark,
    "CSMDialogRun": CSMDialogRun,
    "MultiLinePromptCSM": MultiLinePromptCSM,
    "AudioRecorderCSM": AudioRecorderCSM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddWatermark": "Add Watermark",
    "CSMDialogRun": "CSM Dialog Run",
    "MultiLinePromptCSM": "Multi Line Prompt",
    "AudioRecorderCSM": "MW Audio Recorder",
}