import time
import numpy as np
import onnxruntime
import whisper

# --- 1. Load the combined ONNX model ---
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession("combined_base.en.onnx", sess_options=sess_options)

# --- 2. Load audio and prepare mel spectrogram ---
audio_file = "jfk.wav" # Replace with your audio file path
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
mel = mel.unsqueeze(0).numpy()  # Add batch dimension

# --- 3. Load tokenizer ---
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task="transcribe")

# --- 4. Set inference parameters ---
max_tokens = 512
temperature = 0

# --- 5. ONNX Inference ---
start_time = time.time()

# Initialize decoder input with start of sequence (sot) token
tokens = [tokenizer.sot]

# Loop to generate tokens
for _ in range(max_tokens):
    decoder_input = np.array([tokens], dtype=np.int64)

    # Run the combined ONNX model
    logits, = session.run(["logits"], {"mel": mel, "decoder_input_ids": decoder_input})

    next_token = logits[0, -1].argmax()
    tokens.append(next_token)

    if next_token == tokenizer.eot:
        break

# Decode the generated tokens
transcription = tokenizer.decode(tokens)

# --- 6. Remove special tokens ---
transcription = transcription.replace("<|startoftranscript|>", "")
transcription = transcription.replace("<|notimestamps|>", "")
transcription = transcription.replace("<|endoftext|>", "")

end_time = time.time()
print(f"ONNX Inference Time: {end_time - start_time:.2f} seconds")
print(f"Transcription: {transcription}")
