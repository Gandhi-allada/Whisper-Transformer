import time
import numpy as np
import onnxruntime
import whisper
import sounddevice as sd
import queue

# --- Audio recording parameters ---
sample_rate = 16000  # Adjust if needed based on your microphone
chunk_duration = 10  # Process audio in 10-second chunks

# --- 1. Load the combined ONNX model ---
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession("combined_base.en.onnx", sess_options=sess_options)

# --- 2. Load tokenizer ---
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task="transcribe")

# --- 3. Set inference parameters ---
max_tokens = 512
temperature = 0

# --- Function to process audio chunks ---
def process_audio_chunk(audio_chunk):
    audio_chunk = whisper.pad_or_trim(audio_chunk)
    mel = whisper.log_mel_spectrogram(audio_chunk)
    mel = mel.unsqueeze(0).numpy()

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

    # Remove special tokens
    transcription = transcription.replace("<|startoftranscript|>", "")
    transcription = transcription.replace("<|notimestamps|>", "")
    transcription = transcription.replace("<|endoftext|>", "")

    return transcription

# --- Audio input callback function ---
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Input stream error: {status}")
    q.put(indata.copy().flatten())

# --- Main loop ---
q = queue.Queue()

with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
    print("Listening...")
    audio_buffer = np.array([], dtype=np.float32)
    start_time = time.time()

    while True:
        try:
            # Get audio data from the queue
            audio_buffer = np.concatenate((audio_buffer, q.get()))

            # Process audio in chunks
            if len(audio_buffer) >= int(sample_rate * chunk_duration):
                chunk = audio_buffer[:int(sample_rate * chunk_duration)]
                audio_buffer = audio_buffer[int(sample_rate * chunk_duration):]

                # Process the chunk
                transcription = process_audio_chunk(chunk)
                print(f"Transcription: {transcription}")

        except KeyboardInterrupt:
            break

print("Finished")

