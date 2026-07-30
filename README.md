# llmtts

Prototypes for the conversational voice of **City4All**, a European project built around
a Unity VR experience that helps children understand what disability means in everyday
city life.

The VR experience needed a virtual character children could simply talk to, in Italian,
and hear answer back. This repository is where that capability was worked out before it
reached Unity: each script loads an open-weight language model with Hugging Face
`transformers`, runs a chat loop in the terminal, and reads every reply out loud through
a text-to-speech engine.

It is deliberately a set of side-by-side variants rather than one finished application.
The same loop (*prompt, generate, speak*) is implemented against different models,
different hardware backends and two different TTS engines, so they can be compared on
what actually mattered for the target audience: fluent Italian, answers short enough to
hold a child's attention, and generation fast enough on the hardware at hand to feel like
a conversation.

What these prototypes led to lives in
[unity-voice-agent](https://github.com/Daniel180402/unity-voice-agent): a local server
wrapping llama3, and the Unity C# layer that records a child's voice and plays the
character's reply back inside the scene.

## The variants

| Script | Model | TTS | Target hardware |
| --- | --- | --- | --- |
| `gemma2-it-2b.py` | `google/gemma-2-2b-it` | gTTS (online) | CUDA / CPU |
| `chat_with_gemma2.py` | `google/gemma-2-9b-it` | gTTS (online) | CUDA / CPU |
| `mps_chat_with_gemma2.py` | `google/gemma-2-9b-it` | gTTS (online) | Apple Silicon (MPS) / CPU |
| `llama3-3b.py` | `meta-llama/Llama-3.2-3B` | gTTS (online) | CUDA / CPU |
| `chat_with_llama2.py` | `meta-llama/Llama-2-7b-chat-hf` | pyttsx3 (offline) | CUDA / CPU |
| `chat_with_gpt2.py` | `GroNLP/gpt2-small-italian` | pyttsx3 (offline) | CUDA / CPU |

**Start with `gemma2-it-2b.py`.** It is the most refined version, and the shape the
character's voice settled into. It is small enough to leave headroom for everything else
running alongside it, it applies the model's chat template, and it appends an Italian
instruction asking for a direct answer under 100 words rather than a list of examples.
A spoken answer has no bullet points, and a child will not sit through three paragraphs.
For the same reason it strips bullet-point lines from the reply and trims the text at the
last full stop, so the audio never ends mid-sentence.

`mps_chat_with_gemma2.py` is the Apple Silicon port: same idea, running on the `mps`
backend and playing audio with `afplay`.

The remaining scripts are earlier steps. `chat_with_gpt2.py` uses a small Italian GPT-2
and offline speech, which makes it the only variant that needs neither a gated model nor
an internet connection at runtime. `chat_with_llama2.py` and `llama3-3b.py` feed the raw
user text to the model without a chat template, so their replies read more like text
continuation than conversation.

## The two TTS paths

- **gTTS** calls Google Translate's speech service, so it needs an internet connection.
  It writes `response_full.mp3` next to the script and plays it with the system player
  (`start` on Windows, `afplay` on macOS).
- **pyttsx3** speaks through the voices installed on the operating system. It works
  offline; `chat_with_gpt2.py` looks for an Italian voice and falls back to the default
  one if none is installed.

The choice between the two is a deployment constraint as much as an audio one: gTTS gives
a noticeably more natural Italian voice, but it makes the character depend on network
access every time it speaks.

## Requirements

- Python 3.10+
- A Hugging Face account with access granted to the gated Gemma and Llama repositories
  (request it on the model page before running those scripts)

```bash
pip install torch transformers accelerate sentencepiece gtts pyttsx3
```

For NVIDIA GPUs, install the CUDA build of PyTorch from
[pytorch.org](https://pytorch.org/get-started/locally/) rather than the default wheel.

## Running

```bash
python gemma2-it-2b.py
```

The script asks for your Hugging Face access token, downloads the weights on first run
(cached afterwards in `~/.cache/huggingface`), and then starts the chat loop. Type your
message, wait for the reply, and it will be printed and spoken. Type `exit` to quit.

```
Please enter your Hugging Face access token: hf_...
You can start chatting with the model now (type 'exit' to quit)

You: Cos'è la fotosintesi?
Assistant: La fotosintesi è il processo con cui le piante ...
```

## Notes

- The token is read from stdin every run and never written to disk.
- `response_full.mp3` is overwritten on every reply from the gTTS variants.
- Generation settings (temperature, `top_p`, repetition penalty, `max_new_tokens`) are
  hard-coded near the `model.generate` call in each script, which is the place to tune
  the tone and length of the answers.
