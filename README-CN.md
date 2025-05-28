[中文](README-CN.md)|[English](README.md)

# ComfyUI Node for CSM

CSM（Conversational Speech Model 会话语音模型）, 克隆声音, 多人会话. 暂时支持 2 人会话.

## 📣 更新

[2025-05-29]⚒️: 重新实现核心对话功能. 

[2025-03-18]⚒️: 发布版本 v1.0.0. 

## 预览

- 提示和文本格式必须如下:
```
[S1] Hi, how are you.
[S2] Fine, thank you, and you?
[S1] I'm fine, too.
[S2] What are you planning to do?
```

- 克隆声音会话并保存说话者:
![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_15-38-45.png)

- 加载已保存说话者:
![](https://github.com/billwuhao/ComfyUI_CSM/blob/master/images/2025-03-18_19-01-15.png)


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