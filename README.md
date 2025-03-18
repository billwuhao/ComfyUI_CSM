[中文](README-CN.md)|[English](README.md)

# ComfyUI Node for CSM

CSM (Conversational Speech Model), a model that supports multi-person conversations, voice cloning, and generates speech with corresponding emotional changes based on the emotional changes in the conversation. Unfortunately, it is currently only available in English. This node temporarily supports simultaneous conversations with up to 10 people.

![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_19-01-15.png)

Recording nodes can be interspersed within to create multi-person conversations.

![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_15-38-45.png)

It also supports audio watermark detection (automatic watermark detection) and audio adding encrypted watermarks.

![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_14-43-49.png)

## 📣 Updates

[2025-03-18]⚒️: Released version v1.0.0.

Detailed node usage example workflows: [example_workflows](https://github.com/billwuhao/ComfyUI_CSM/blob/master/example_workflows)

The `prompt` must be preceded by a number from `0~9`, and separated by a `:` or `：`.  Prompts and audio need to correspond one-to-one, for example, `prompt1` corresponds to `audio1`.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_CSM.git
cd ComfyUI_CSM
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- [csm-1b](https://huggingface.co/sesame/csm-1b/tree/main): Download `config.json` and `model.safetensors` and place them in the `ComfyUI/models/TTS/csm-1b` directory.

- [moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16/tree/main): Download `tokenizer-e351c8d8-checkpoint125.safetensors` and place it in the `ComfyUI/models/TTS/moshiko-pytorch-bf16` directory.

- [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration): Download all models and place them in the `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` directory.

- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main): Download everything except the `original` directory and place it in the `ComfyUI\models\LLM\Llama-3.2-1B` directory.

## Acknowledgements

[csm](https://github.com/SesameAILabs/csm)

Thanks to the SesameAILabs team for their excellent work 👍.
