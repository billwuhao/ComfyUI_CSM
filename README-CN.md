[中文](README-CN.md)|[English](README.md)

# ComfyUI Node for CSM

CSM（Conversational Speech Model 会话语音模型）, 多人会话, 克隆声音, 然后可根据会话中语音的情绪变化等, 生成相应情绪变化的语音的模型. 遗憾的是目前只有英文可用. 该节点, 暂时支持同时 10 人会话.

![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_19-01-15.png)

可以将录音节点穿插在其中, 进行多人会话.

![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_15-38-45.png)

## 📣 更新

[2025-03-18]⚒️: 发布版本 v1.0.0. 

节点使用方法详解示例工作流: [example_workflows](https://github.com/billwuhao/ComfyUI_CSM/blob/master/example_workflows)

提示词 `prompt` 前面必须是 `0~9` 的数字, 并且有 `:` 或 `：` 号分隔. 提示词和音频需要一一对应, 例如 `prompt1` 对应 `audio1`.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_CSM.git
cd ComfyUI_CSM
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- [csm-1b](https://huggingface.co/sesame/csm-1b/tree/main): `config.json` 和 `model.safetensors` 下载放到 `ComfyUI/models/TTS/csm-1b` 目录下.

- [moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16/tree/main): `tokenizer-e351c8d8-checkpoint125.safetensors` 下载放到 `ComfyUI/models/TTS/moshiko-pytorch-bf16` 目录下.

- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main): 除了 `original` 目录, 其他全部下载放到 `ComfyUI\models\LLM\Llama-3.2-1B` 目录下.

## 鸣谢

[csm](https://github.com/SesameAILabs/csm)

感谢 SesameAILabs 团队的卓越的工作👍.