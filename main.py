import asyncio
from dotenv import load_dotenv
import logging
from audio import SpeechToText, TextToSpeech
import queue
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
        self.stt_queue: queue.Queue = queue.Queue()
        self.tts_queue: queue.Queue = queue.Queue()
        self.stt: SpeechToText = SpeechToText(output_queue=self.stt_queue, model_size="small")
        self.tts: TextToSpeech = TextToSpeech(workspace=workspace)
        self.gemini = Gemini()

    async def run(self):
        threading.Thread(target=self.stt.listen, daemon=True).start()
        while True:
            if not self.stt_queue.empty():
                self.tts.speak(self.gemini.get_response(self.stt_queue.get()))

            await asyncio.sleep(0.1)

if __name__ == "__main__":
    load_dotenv()

    with new_workspace() as workspace:
        core: Core = Core(workspace=workspace)
        asyncio.run(core.run())
