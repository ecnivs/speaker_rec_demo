import asyncio
from dotenv import load_dotenv
import logging
from audio import SpeechToText
import queue
import threading

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    force=True)

class Core:
    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.queue: queue.Queue = queue.Queue()
        self.stt: SpeechToText = SpeechToText(output_queue=self.queue, model_size="small")

    async def run(self):
        threading.Thread(target=self.stt.listen, daemon=True).start()
        while True:
            if not self.queue.empty():
                self.logger.info(f"You: {self.queue.get()}")
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    load_dotenv()

    core = Core()
    asyncio.run(core.run())
