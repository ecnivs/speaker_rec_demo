import queue
from dotenv import load_dotenv
import logging
from audio import SpeechToText, TextToSpeech
import threading
import shutil
from contextlib import contextmanager
import tempfile
from response import Llm
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
        self.llm: Llm = Llm()

    def _process_queue(self):
        if not self.tts_queue.empty():
            self.tts.play_wav(self.tts_queue.get())
        elif not self.tts.is_playing:
            with self.stt.condition:
                self.stt.condition.notify()
                self.stt.pause_listening = False

    def _thread(self, target, args = None):
        if args:
            threading.Thread(target=target, args=args, daemon=True).start()
        else:
            threading.Thread(target=target, daemon=True).start()

    def run(self):
        try:
            self._thread(target=self.stt.listen)

            while True:
                with self.stt.lock:
                    if self.stt.query:
                        (speaker, query), = self.stt.query.items()

                        self.stt.pause_listening = True
                        response = self.llm.get_response(speaker=speaker, query=query)
                        self.logger.info(response)

                        lang = response["language"].lower()
                        try:
                            if ((self.tts.voices_dir) / f"{lang}.wav").exists():
                                self._thread(target=self.tts.speak_local, args=(response["response"], lang))
                            else:
                                self._thread(target=self.tts.speak, args=(response["transcripted_response"], lang))
                        except Exception as e:
                            self._thread(target=self.tts.speak, args=(response["transcripted_response"], lang))

                    self.stt.query = None

                self._process_queue()
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
