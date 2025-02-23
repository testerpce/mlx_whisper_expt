import numpy as np
import sounddevice as sd
import soundfile as sf
import wave
import threading
import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Parameters
samplerate = 16000  # Sample rate in Hz
channels = 1        # Number of audio channels
dtype = 'int16'     # Data type for recording
#filename = 'recording.wav'  # Output file name

# Buffer to hold recorded audio
audio_buffer = []

# Flag to control recording
is_recording = threading.Event()

def audio_callback(indata, frames, time, status):
    """Callback function to capture audio data."""
    if status:
        print(f"Recording error: {status}")
    if is_recording.is_set():
        audio_buffer.append(indata.copy())

def start_recording():
    """Start the audio recording."""
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype=dtype, callback=audio_callback):
        print("Recording... Press Enter to stop.")
        is_recording.set()
        input()  # Wait for user to press Enter
        is_recording.clear()
        print("Recording stopped.")

def save_recording(filename):
    """Save the recorded audio to a WAV file."""
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(np.dtype(dtype).itemsize)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        print(f"Recording saved as '{filename}'.")
    else:
        print("No audio data to save.")


def play_recording(filename):
    """
    Play an audio recording from the specified WAV file.

    Parameters:
    filename (str): The path to the WAV file to be played.
    """
    try:
        # Load audio data from the file
        data, samplerate = sf.read(filename, dtype='int16')

        # Play the audio data
        print(f"Playing '{filename}'...")
        sd.play(data, samplerate)
        sd.wait()  # Wait until playback is finished
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred while playing the file: {e}")

def transcribe_audio(filename, language=None):
    """
    Transcribe an audio recording from the specified file using OpenAI's Whisper model.

    Parameters:
    filename (str): The path to the audio file to be transcribed.
    language (str, optional): The language code of the audio (e.g., 'en' for English, 'fr' for French).
                              If not provided, the model will use its default settings.

    Returns:
    str: The transcribed text.
    """
    # Check if a GPU is available and set the device and data type accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the Whisper model and processor
    model_id = "openai/whisper-large-v3"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)

    # Initialize the ASR pipeline with or without language settings
    if language:
        # Generate forced decoder ids for the specified language
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        generate_kwargs = {"forced_decoder_ids": forced_decoder_ids}
    else:
        generate_kwargs = {}

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # Process audio in 30-second chunks
        device=device,
        generate_kwargs=generate_kwargs,
    )

    # Load audio data from the file
    audio_input, sample_rate = sf.read(filename)

    # Ensure the audio is in the correct format
    if sample_rate != 16000:
        raise ValueError("The Whisper model expects audio with a sample rate of 16 kHz.")

    # Transcribe the audio
    result = asr_pipeline(audio_input)

    return result["text"]

def main():
    args = get_args()
    input("Press Enter to start recording...")
    start_recording()
    save_recording(args.output_file)
    play_recording(args.output_file)
    if args.language:
        transcription = transcribe_audio(args.output_file, args.language)
    else:
        transcription = transcribe_audio(args.output_file)

    # Print the transcription
    print("Transcription:", transcription)


def get_args():
    """
    Get command line arguments using argparse.

    Returns:
        str: The output file name.
    """
    parser = argparse.ArgumentParser(description='Record audio to a file')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file name')
    parser.add_argument('-l', '--language', type=str, help='Language to transcribe')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
