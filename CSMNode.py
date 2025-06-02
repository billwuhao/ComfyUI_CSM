from dataclasses import dataclass
from typing import List, Tuple
import torch
import torchaudio
import os
import json
import safetensors.torch

# from huggingface_hub import hf_hub_download
from .models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import folder_paths


models_dir = folder_paths.models_dir
speakers_dir = os.path.join(models_dir, "TTS", "speakers", "dialogue_speakers")

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor
    
class Generator:
    def __init__(
        self,
        model: Model,
        device: str = "cuda",
    ):
        self.model = model
        self.model.setup_caches(1)
        self.device = device

        self.text_tokenizer = self.load_llama3_tokenizer()

        mimi = self.load_mimi()
        self.audio_tokenizer = mimi
        self.sample_rate = mimi.sample_rate

    def clean_memory(self):
        self.model = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
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

        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
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
        audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
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
        texts: List[str],
        speakers: List[int],
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:

        all_generated_audio = []

        for i in range(len(texts)):
            current_text = texts[i]
            current_speaker = speakers[i]

            self.model.reset_caches()

            max_audio_frames = int(max_audio_length_ms / 80)
            tokens, tokens_mask = [], []
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

            gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(current_text, current_speaker)
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
                # Potentially skip this audio or handle error differently if one segment is too long
                print(f"Warning: Input for text '{current_text}' is too long and will be skipped.")
                continue # Or raise ValueError

            for _ in range(max_audio_frames):
                sample = self.model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break  # eos

                samples.append(sample)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
            
            if samples:
                audio_segment = self.audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
                all_generated_audio.append(audio_segment)
            else:
                print(f"Warning: No audio samples generated for text '{current_text}'.")

        if not all_generated_audio:
            return torch.empty(0).to(self.device) 

        final_audio = torch.cat(all_generated_audio, dim=0) 

        return final_audio


class MultiLineText:
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
    RETURN_NAMES = ("text",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


MODEL_CACHE = None
class CSMDialogRun:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": (["model.safetensors", 
                               "chinese_model.safetensors", 
                               "model_bf16.safetensors", 
                               "model_fp16.safetensors", 
                               "model_int8.safetensors", 
                               "model_uint8.safetensors",
                               ], 
                               {"default": "model.safetensors"}
                    ),
                    "text": ("STRING", {"forceInput": True}),
                    "prompt": ("STRING",  {
                        "multiline": True, 
                        "default": ""}),
                    "audio_s1": ("AUDIO",),
                    "audio_s2": ("AUDIO",),
                    "max_audio_length_ms": ("INT", {
                        "default": 2000,
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
                    "top_k": ("INT", {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1
                    }),
                    "save_speakers": ("BOOLEAN", {"default": True}),
                    "speakers_id": ("STRING", {"default": "A_and_B"}),
                    "unload_model": ("BOOLEAN", {
                            "default": True,
                            "tooltip": "Unload model from memory after use"
                    }),
                },
                "optional": {
                }
        }

    CATEGORY = "🎤MW/MW-CSM"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"

    def run(self, 
            model,
            text, 
            unload_model, 
            prompt, 
            audio_s1,
            audio_s2,
            max_audio_length_ms=90_000,
            temperature=0.9,
            top_k=50,
            save_speakers=True,
            speakers_id="A_and_B",
            ):

        global MODEL_CACHE
        if MODEL_CACHE is None or self.model_name != model:
            self.model_name = model
            csm_1b_path = os.path.join(models_dir, "TTS", "csm-1b")
            config_path = os.path.join(csm_1b_path, "config.json")

            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            config_args = config["args"]
            configs = ModelArgs(backbone_flavor = config_args["backbone_flavor"], 
                                decoder_flavor = config_args["decoder_flavor"], 
                                text_vocab_size = config_args["text_vocab_size"], 
                                audio_vocab_size = config_args["audio_vocab_size"], 
                                audio_num_codebooks = config_args["audio_num_codebooks"])

            MODEL_CACHE = Model(configs)
            safetensors_file_path = os.path.join(csm_1b_path, model)
            state_dict = safetensors.torch.load_file(safetensors_file_path, device="cpu")
            MODEL_CACHE.load_state_dict(state_dict)
            MODEL_CACHE.to(device=self.device)
            MODEL_CACHE.eval()

        generator = Generator(MODEL_CACHE, device=self.device)
        prompt = prompt.strip()
        speakers, texts = self.get_speaker_text(text.strip())

        if len(speakers) != len(texts):
            raise ValueError("The number of speakers and texts in the prompt must be the same.")

        sr = generator.sample_rate

        if not prompt:
            raise ValueError("Prompt can't empty: [S1]... [S2]...")

        p_speakers, p_texts = self.get_speaker_text(prompt)
        if len(p_speakers) != len(p_texts):
            raise ValueError("The number of speakers and texts in the prompt must be the same.")
        if len(p_speakers) == 0:
            raise ValueError("Prompt: [S1]... [S2]...")

        segments = []
        for s, t in zip(p_speakers, p_texts):
            if s == 0:
                segments.append(Segment(speaker=0, text=t, audio=self.get_audio_tensor(audio_s1, generator.sample_rate)))
            elif s == 1:
                segments.append(Segment(speaker=1, text=t, audio=self.get_audio_tensor(audio_s2, generator.sample_rate)))
        
        audio = generator.generate(
            texts=texts,
            speakers=speakers,
            context=segments,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=top_k,
        )
            
        if save_speakers:
            if speakers_id.strip() == "":
                raise ValueError("Speakers ID is empty.")

            if not os.path.exists(speakers_dir):
                os.makedirs(speakers_dir)

            audio_s1_path = os.path.join(speakers_dir, f"{speakers_id}_1.wav")
            torchaudio.save(audio_s1_path, audio_s1["waveform"].squeeze(0), audio_s1["sample_rate"])

            audio_s2_path = os.path.join(speakers_dir, f"{speakers_id}_2.wav")
            torchaudio.save(audio_s2_path, audio_s2["waveform"].squeeze(0), audio_s2["sample_rate"])

            text_path = os.path.join(speakers_dir, f"{speakers_id}.txt")
            
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(prompt)

        if unload_model:
            generator.clean_memory()
            generator = None
            MODEL_CACHE = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return ({"waveform": audio.unsqueeze(0).unsqueeze(0).cpu(), "sample_rate": sr},)

    def get_audio_tensor(self, audio, sample_rate):
        audio_tensor = audio["waveform"].squeeze(0).mean(dim=0)
        orig_freq = int(audio["sample_rate"])

        audio_tensor = torchaudio.functional.resample(
                                                    audio_tensor.squeeze(0), 
                                                    orig_freq=orig_freq, 
                                                    new_freq=sample_rate
                                                )
        return audio_tensor

    def get_speaker_text(self, text):
        import re
        
        pattern = r'(\[s?S?1\]|\[s?S?2\])\s*([\s\S]*?)(?=\[s?S?[12]\]|$)'
        matches = re.findall(pattern, text)
        
        labels = []
        contents = []
        
        for label, content in matches:
            labels.append(label)
            contents.append(content)
        
        numeric_labels = [
            0 if i.lower() == '[s1]' else 1 for i in labels
        ]
        
        return (numeric_labels, contents)
    

from typing import List, Optional, Union

def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths


def get_speakers():
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".txt"], relative_path=True)
    return speakers


class CSMSpeakersPreview:
    def __init__(self):
        self.speakers_dir = speakers_dir
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("STRING", "AUDIO", "AUDIO",)
    RETURN_NAMES = ("prompt", "audio_s1", "audio_s2",)
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-CSM"

    def preview(self, speaker):
        text_path = os.path.join(self.speakers_dir, speaker)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        audio_s1_path = text_path.replace(".txt", "_1.wav")
        waveform, sample_rate = torchaudio.load(audio_s1_path)
        waveform = waveform.unsqueeze(0)
        output_audio_s1 = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        audio_s2_path = text_path.replace(".txt", "_2.wav")
        waveform, sample_rate = torchaudio.load(audio_s2_path)
        waveform = waveform.unsqueeze(0)
        output_audio_s2 = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        return (text, output_audio_s1, output_audio_s2)


NODE_CLASS_MAPPINGS = {
    "CSMDialogRun": CSMDialogRun,
    "CSMSpeakersPreview": CSMSpeakersPreview,
    "MultiLineText": MultiLineText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSMDialogRun": "CSM Dialog Run",
    "CSMSpeakersPreview": "Speakers Preview",
    "MultiLineText": "Multi Line Text",
}