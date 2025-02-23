# Audio Transcription Experiment using Whisper
This code is a small experiment to record audio and transcribe it using Whisper. Currently it doesn't use mlx. It just uses huggingface models to transcribe.

## Running the Code
To run the code, use the following command:
```bash
python audio_keyboard.py -o Recording/record_1.wav
```
Press Enter to start recording audio. Once you are done. Press Enter to stop. It will first playback the audio and then transcribe it using whisper and print it.
This will record audio, save it, and print the transcribed version.
Forcing Language Output
To force the model to output in a specific language, use the -e flag followed by the language code. For example, to output in Kannada:
```bash
python audio_keyboard.py -o Recording/record_1.wav -e kn
```
This will record audio, save it, and print the transcribed version in Kannada.
