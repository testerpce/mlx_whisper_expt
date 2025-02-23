# Audio Transcription Experiment using Whisper
This code is a small experiment to record audio and transcribe it using Whisper.

## Running the Code
To run the code, use the following command:
```bash
python audio_keyboard.py -o Recording/record_1.wav


This will record audio, save it, and print the transcribed version.
Forcing Language Output
To force the model to output in a specific language, use the -e flag followed by the language code. For example, to output in Kannada:
```bash
python audio_keyboard.py -o Recording/record_1.wav -e kn

This will record audio, save it, and print the transcribed version in Kannada.
