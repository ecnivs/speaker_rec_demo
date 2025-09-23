from google.genai import Client, types
import wave
import os
import logging
import time
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
import torch
import queue
from pathlib import Path
import re

class TextToSpeech:
    def __init__(self, workspace, output_queue: queue.Queue[str]):
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.client: Client = Client()
        self.workspace = workspace
        self.queue: queue.Queue[str] = output_queue

        self.gemini_voice: str = "zephyr"
        self.gemini_model: str = "gemini-2.5-flash-preview-tts"

        self.coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(model_name=self.coqui_model, progress_bar=True).to(self.device)

        self.voices_dir: Path = Path(".voices")
        self.voices_dir.mkdir(exist_ok=True)

    def play_wav(self, path: str):
        try:
            audio, samplerate = sf.read(path)
            sd.play(audio, samplerate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Error playing {path}: {e}")

    def speak_local(self, text: str, language: str):
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        for sentence in sentences:
            if not sentence:
                continue
            path = os.path.join(self.workspace, f'{time.time_ns()}_speech.wav')
            self.tts.tts_to_file(
                sentence,
                file_path=path,
                speaker_wav=self.voices_dir / f"{language}.wav",
                language=language
            )
            self.queue.put(str(path))

    def speak(self, transcript: str, language: str):
        try:
            if not transcript:
                raise ValueError("Transcript must be a non-empty string")
            self.logger.info(transcript)

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.gemini_voice,
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

            path = (self.workspace / f'{time.time_ns()}_speech.wav')
            language_wav = (self.voices_dir / f"{language}.wav")
            if language_wav.exists():
                info = sf.info(language_wav)
                if (info.frames / info.samplerate) <= 3:
                    path = language_wav
            if not language_wav.exists():
                path = language_wav

            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(30000)
                wf.writeframes(data)

            self.queue.put(str(path))

        except Exception as e:
            raise Exception(f"Failed to generate audio: {e}")
