from pathlib import Path

from agents.hri import HRI_Agent, HRI_Agent_Config
from transcript import TerminalTranscript


WORKSPACE_ROOT = Path(__file__).resolve().parent
TRANSCRIPT_PATH = WORKSPACE_ROOT / "experiments" / "record.txt"


class PrefMem:
    def __init__(self):
        self.hri_agent = HRI_Agent(HRI_Agent_Config())

    def start(self):
        print(f"Starting PrefMem...")
        self.hri_agent.get_response()




if __name__ == "__main__":
    with TerminalTranscript(TRANSCRIPT_PATH):
        pref_mem = PrefMem()
        pref_mem.start()
