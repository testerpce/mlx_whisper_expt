import speech_recognition as sr
# print(s_r.Microphone.list_microphone_names())
# mic = s_r.Microphone(device_index=1)
# print(mic)
# my_mic = s_r.Microphone()
# print(my_mic)
# Initialize the recognizer
r = sr.Recognizer()

# List all available microphone devices
print(sr.Microphone.list_microphone_names())

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Say something!")
    # Adjust for ambient noise if necessary
    # r.adjust_for_ambient_noise(source)
    # Listen for the first phrase and extract it into audio data
    audio = r.listen(source)

# Recognize speech using Google Speech Recognition
try:
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")