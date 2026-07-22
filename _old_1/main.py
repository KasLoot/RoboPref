import argparse
from pathlib import Path

from agents.hri import HRI_Agent, HRI_Agent_Config
from transcript import TerminalTranscript


WORKSPACE_ROOT = Path(__file__).resolve().parent
TRANSCRIPT_PATH = WORKSPACE_ROOT / "experiments" / "record.txt"


class PrefMem:
    def __init__(self, config: HRI_Agent_Config | None = None):
        self.hri_agent = HRI_Agent(config or HRI_Agent_Config())

    def start(self):
        print(f"Starting PrefMem...")
        self.hri_agent.get_response()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RoboPref recorded-scene episode.")
    parser.add_argument("--dataset", help="Episode directory containing numbered image frames.")
    parser.add_argument("--memory-store", help="Participant-specific preference JSON path.")
    parser.add_argument("--user-id", default="default", help="Non-identifying participant code.")
    parser.add_argument("--transcript", help="Participant/session transcript output path.")
    args = parser.parse_args()

    config = HRI_Agent_Config()
    if args.dataset:
        config.dataset_path = args.dataset
    if args.memory_store:
        config.memory_store_path = args.memory_store
    config.user_id = args.user_id
    transcript_path = Path(args.transcript) if args.transcript else TRANSCRIPT_PATH

    with TerminalTranscript(transcript_path):
        pref_mem = PrefMem(config)
        pref_mem.start()
