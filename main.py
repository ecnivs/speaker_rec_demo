import asyncio
import queue
from dotenv import load_dotenv
import logging
from audio import SpeechToText, TextToSpeech
import threading
import shutil
from contextlib import contextmanager
import tempfile
from response import Gemini

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
        self.lock: threading.Lock = threading.Lock()
        self.stt: SpeechToText = SpeechToText(core=self, model_size="small")
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts: TextToSpeech = TextToSpeech(workspace=workspace, output_queue=self.tts_queue)
        self.gemini: Gemini = Gemini()
        self.query = None

    def _process_queue(self):
        if not self.tts_queue.empty():
            self.tts.play_wav(self.tts_queue.get())

    async def run(self):
        threading.Thread(target=self.stt.listen, daemon=True).start()
        while True:
            self._process_queue()
            with self.lock:
                if self.query:
                    response = self.gemini.get_response(self.query)
                    self.logger.info(response)
                    if response.get("language") == "en":
                        self.tts.speak_local(response["response"])
                    else:
                        self.tts.speak(transcript=response["transcripted_response"])

                self.query = None
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    load_dotenv()

    with new_workspace() as workspace:
        core: Core = Core(workspace=workspace)
        asyncio.run(core.run())
