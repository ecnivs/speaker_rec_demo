import queue
from dotenv import load_dotenv
import logging
from audio import SpeechToText, TextToSpeech
import threading
import shutil
from contextlib import contextmanager
import tempfile
from response import Gemini
import time
from pathlib import Path

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    force=True)

# -------------------------------
# Temporary Workspace
# -------------------------------
@contextmanager
def new_workspace():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class Core:
    def __init__(self, workspace) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.stt: SpeechToText = SpeechToText(model_size="small")
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts: TextToSpeech = TextToSpeech(workspace=workspace, output_queue=self.tts_queue)
        self.gemini: Gemini = Gemini()

    def _process_queue(self):
        if not self.tts_queue.empty():
            self.tts.play_wav(self.tts_queue.get())
        else:
            with self.stt.condition:
                self.stt.pause_listening = False
                self.stt.condition.notify()

    def run(self):
        try:
            self.speech_thread = threading.Thread(target=self.stt.listen, daemon=True).start()
            while True:
                self._process_queue()
                with self.stt.lock:
                    if self.stt.query:
                        self.stt.pause_listening = True
                        response = self.gemini.get_response(self.stt.query)
                        self.logger.info(response)

                        lang = response["language"]
                        try:
                            if ((self.tts.voices_dir) / f"{lang}.wav").exists():
                                self.logger.info("Using Local TTS")
                                self.tts.speak_local(text=response["response"], language=lang)
                            else:
                                self.logger.info("Using Gemini TTS")
                                self.tts.speak(transcript=response["transcripted_response"], language=lang)
                        except Exception as e:
                            self.logger.info(f"Fallback to Gemini TTS: {e}")
                            self.tts.speak(transcript=response["transcripted_response"], language=lang)

                    self.stt.query = None

                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        except Exception as e:
            self.logger.critical(f"Error: {e}")

if __name__ == "__main__":
    load_dotenv()

    with new_workspace() as workspace:
        core: Core = Core(workspace=Path(workspace))
        core.run()
