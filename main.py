import openai
from agents.hri import HRI_Agent, HRI_Agent_Config

class PrefMem:
    def __init__(self):
        self.hri_agent = HRI_Agent(HRI_Agent_Config())

    def start(self):
        print(f"Starting PrefMem...")
        self.hri_agent.get_response()




if __name__ == "__main__":
    pref_mem = PrefMem()
    pref_mem.start()