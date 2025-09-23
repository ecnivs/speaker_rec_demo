import threading
import numpy as np
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import logging
import queue
import time
from typing import List, Optional, Any
import webrtcvad
from pyannote.audio import Model, Inference
import os
import torch
from pathlib import Path

class SpeechToText(threading.Thread):
    def __init__(self, model_size: str ="small") -> None:
        super().__init__(daemon=True)
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.query = None
        self.pause_listening: bool = True

        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("HF_API_KEY"))
        self.inference = Inference(self.model, window="whole")
        self.speakers_dir = Path(".speakers")
        self.embeddings = self._load_embeddings()

        self.sample_rate: int = 16000
        self.chunk_duration: float = 3
        self.overlap_duration: float = 0.15
        self.chunk_size: int = int(self.chunk_duration * self.sample_rate)
        self.overlap_size: int = int(self.overlap_duration * self.sample_rate)

        self.buffer: List[float] = []
        self.condition:threading.Condition = threading.Condition()
        self.lock: threading.Lock = threading.Lock()
        self.running: threading.Event = threading.Event()
        self.running.set()

        self.whisper_model: WhisperModel = WhisperModel(model_size, device="cuda")
        self.vad: webrtcvad.Vad = webrtcvad.Vad(2)

        self.transcription_queue: queue.Queue[np.ndarray] = queue.Queue()
        threading.Thread(target=self._transcription_worker, daemon=True).start()

    def _load_embeddings(self):
        return {f.stem: self.inference(str(f)) for f in self.speakers_dir.glob("*.wav")}

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Optional[sd.CallbackFlags]) -> None:
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        if not self.pause_listening:
            self.buffer.extend(indata[:, 0])

    def listen(self) -> None:
        self.logger.info("Listening...")

        with sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024
        ):
            while self.running.is_set():

                if self.pause_listening:
                    with self.condition:
                        while self.pause_listening:
                            self.condition.wait()
                        self.buffer.clear()

                if len(self.buffer) >= self.chunk_size:
                    chunk = np.array(self.buffer[:self.chunk_size], dtype=np.float32)
                    self.buffer = self.buffer[self.chunk_size - self.overlap_size:]

                    pcm_data: bytes = (chunk * 32768).astype(np.int16).tobytes()
                    frame_duration_ms: int = 30
                    frame_size: int = int(self.sample_rate * frame_duration_ms / 1000)

                    speech_frames: int = 0
                    for start in range(0, len(chunk), frame_size):
                        frame: bytes = pcm_data[start*2:(start+frame_size)*2]
                        if len(frame) < frame_size * 2:
                            break
                        if self.vad.is_speech(frame, sample_rate=self.sample_rate):
                            speech_frames += 1

                    if speech_frames < 2:
                        continue

                    self.transcription_queue.put(chunk)
                else:
                    time.sleep(0.01)

    def _get_best_speaker(self, chunk: np.ndarray, threshold: float) -> str | None:
        sample = {"waveform": torch.tensor(chunk).unsqueeze(0), "sample_rate": self.sample_rate}
        emb = self.inference(sample)

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy().flatten()

        best_speaker = None
        best_sim = -1.0
        for name, ref in self.embeddings.items():
            if isinstance(ref, torch.Tensor):
                ref_vec = ref.detach().cpu().numpy().flatten()
            else:
                ref_vec = ref.flatten()

            sim = np.dot(ref_vec, emb) / (np.linalg.norm(ref_vec) * np.linalg.norm(emb))

            if sim > best_sim:
                best_sim = sim
                best_speaker = name

        self.logger.info(f"Similarity with {best_speaker}: {best_sim:.3f}")

        if best_sim >= threshold:
            return best_speaker
        return None

    def _transcription_worker(self) -> None:
        last_text = ""
        while self.running.is_set() or not self.transcription_queue.empty():
            try:
                chunk = self.transcription_queue.get(timeout=0.1)

                speaker = self._get_best_speaker(chunk, threshold=0.2)
                if speaker is None:
                    self.logger.info("Ignored: speaker does not match any known speaker")
                    continue

                segments, info = self.whisper_model.transcribe(chunk, beam_size=5)
                for segment in segments:
                    text = segment.text.strip()
                    if not text or text == last_text or text.lower() == "you" or info.language == "nn":
                        continue

                    with self.lock:
                        self.query = { speaker.capitalize(): text }
                    self.logger.info(f"{speaker.capitalize()}: {text}")
                    last_text = text
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error during transcription: {e}")

    def stop(self) -> None:
        self.running.clear()
        self.logger.info("Stopping SpeechToText thread")
