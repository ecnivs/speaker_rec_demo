from google.genai import Client, types
import wave
import os
import logging
import time
import sounddevice as sd
import soundfile as sf

class TextToSpeech:
    def __init__(self, workspace):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.client: Client = Client()
        self.workspace = workspace
        self.voice: str = "Kore"

    def _save_to_wav(self, pcm):
        path = os.path.join(self.workspace, f'{time.time_ns()}_speech.wav')
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(30000)
            wf.writeframes(pcm)

        self._play_wav(path)

    def _play_wav(self, path: str):
        try:
            audio, samplerate = sf.read(path)
            sd.play(audio, samplerate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Error playing {path}: {e}")

    def speak(self, transcript: str):
        try:
            if not transcript:
                raise ValueError("Transcript must be a non-empty string")

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=transcript,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice,
                            )
                        )
                    ),
                )
            )

            if not response.candidates:
                raise ValueError("No candidates returned from TTS model")
            if not response.candidates[0].content.parts:
                raise ValueError("No content parts returned in response")

            data = response.candidates[0].content.parts[0].inline_data.data
            if not data:
                raise ValueError("No audio data found in response")

            self._save_to_wav(data)

        except Exception as e:
            raise Exception(f"Failed to generate audio: {e}")
