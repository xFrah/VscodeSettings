import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

audio_files = os.listdir("audio")
for audio_file in audio_files:
    result = pipe(f"audio/{audio_file}")
    # put result in new folder with same name as audio file 

result = pipe("Audio WhatsApp 2024-02-23 ore 14.31.22_fdd07d1f.waptt.opus")
print(result["text"])
