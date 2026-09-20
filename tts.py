"""
tts.py
On-device text-to-speech wrapper for SignSpeak AI.

Uses pyttsx3 (fully offline, OS-native voices) as a lightweight default.
Swap the `speak` implementation for a Qualcomm AI Hub-optimized TTS model
(e.g. a distilled VITS/FastSpeech variant) for the final NPU-accelerated
submission.
"""

import pyttsx3


class OfflineTTS:
    def __init__(self, rate: int = 165, volume: float = 1.0, voice_index=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        if voice_index is not None:
            voices = self.engine.getProperty("voices")
            if 0 <= voice_index < len(voices):
                self.engine.setProperty("voice", voices[voice_index].id)

    def speak(self, text: str):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()

    def list_voices(self):
        for i, voice in enumerate(self.engine.getProperty("voices")):
            print(f"[{i}] {voice.name} ({voice.languages})")


if __name__ == "__main__":
    tts = OfflineTTS()
    tts.list_voices()
    tts.speak("Sign Speak A I is ready.")
