import time
import numpy as np
import onnxruntime
import whisper
import sounddevice as sd
import queue
import speech_recognition as sr

# --- Audio recording parameters ---
sample_rate = 16000
chunk_duration = 5  # Process audio in smaller chunks
silence_threshold = 15  # Seconds of silence before stopping
keyword = "start"  # Keyword to start recording

# --- 1. Load ONNX models ---
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_encoder = onnxruntime.InferenceSession("encoder_base.en.onnx", sess_options=sess_options)
sess_decoder = onnxruntime.InferenceSession("decoder_base.en.onnx", sess_options=sess_options)

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

    # Encode the audio
    encoder_output, = sess_encoder.run(["encoder_output"], {"mel": mel})

    # Initialize decoder input with start of sequence (sot) token
    tokens = [tokenizer.sot]

    # Loop to generate tokens
    for _ in range(max_tokens):
        # Prepare decoder input
        decoder_input = np.array([tokens], dtype=np.int64)

        # Run the decoder
        logits, = sess_decoder.run(["logits"], {"tokens": decoder_input, "encoder_output": encoder_output})

        # Sample the next token (greedy decoding for now)
        next_token = logits[0, -1].argmax()

        # Append the token to the sequence
        tokens.append(next_token)

        # Stop if end-of-sequence (eot) token is generated
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

# --- Function to check for keyword ---
def listen_for_keyword(source, keyword):
    print("Waiting for keyword...")
    while True:
        audio_data = r.listen(source)
        try:
            text = r.recognize_google(audio_data)
            print(f"You said: {text}")
            if keyword.lower() in text.lower():
                return True
        except sr.UnknownValueError:
            print("...")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# --- Main loop ---
q = queue.Queue()
recording = False
silence_start = None
r = sr.Recognizer()

with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
    print("Listening for keyword...")

    with sr.Microphone() as source:
        listen_for_keyword(source, keyword) 
        print("Keyword detected! Recording started.")
        recording = True
        audio_buffer = np.array([], dtype=np.float32)

        while True:
            if not recording:
                break

            try:
                # Get audio data from the queue
                audio_buffer = np.concatenate((audio_buffer, q.get()))

                # Process audio in chunks
                if len(audio_buffer) >= int(sample_rate * chunk_duration):
                    chunk = audio_buffer[:int(sample_rate * chunk_duration)]
                    audio_buffer = audio_buffer[int(sample_rate * chunk_duration):]

                    # Check for silence
                    if np.max(np.abs(chunk)) < 0.02:  # Adjust threshold if needed
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > silence_threshold:
                            print("Silence detected, stopping recording.")
                            recording = False
                            break
                    else:
                        silence_start = None 

                    # Process the chunk if still recording
                    if recording:
                        transcription = process_audio_chunk(chunk)
                        print(f"Transcription: {transcription}")

            except KeyboardInterrupt:
                break

print("Finished")
