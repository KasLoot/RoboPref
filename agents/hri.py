import json

from ollama import chat
from colorama import Fore, Back, Style

from agents.planner import Planner_Agent, Planner_Agent_Config
from agents.vlidator import Validator_Agent, Validator_Agent_Config



class HRI_Agent_Config:
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    system_prompt_path: str = "/home/yuxin/workspace/RoboPref/prompt/hri/prompt-v1.md"
    initial_frame_path: str = "/home/yuxin/workspace/RoboPref/dataset/v3/1.png"
    final_frame_path: str = "/home/yuxin/workspace/RoboPref/dataset/v3/12.png"


class HRI_Agent:
    def __init__(self, config: HRI_Agent_Config):
        self.config = config
        self.host = config.host
        self.base_url = config.base_url
        self.model = config.model
        self.initial_frame_path = config.initial_frame_path
        self.final_frame_path = config.final_frame_path
        with open(config.system_prompt_path, "r") as f:
            self.system_prompt = f.read()
        self.conversation = []
        self.planner_agent = Planner_Agent(Planner_Agent_Config())
        self.validator_agent = Validator_Agent(Validator_Agent_Config())

    def get_response(self):
        print("\n--- Start of Conversation ---\n")
        user_query = input("User: ")
        if not user_query:
            user_query = "Stacked vertically the blocks from the right mat on the left mat. RGB sequence from bottom to top."
            print(f"Using default user query: {user_query}")
        self.conversation.extend([
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': user_query,
                'images': [self.initial_frame_path]
            }
        ])

        


        while True:
            stream = chat(
                model=self.model,
                messages=self.conversation,
                think=False,
                stream=True,
                format = "json",
                options = {
                    "temperature": 0.0
                }
            )
            in_thinking = False
            content = ''
            thinking = ''
            for chunk in stream:
                if chunk.message.thinking:
                    if not in_thinking:
                        in_thinking = True
                        print(Fore.YELLOW + 'Thinking:\n', end='', flush=True)
                    print(Fore.YELLOW + chunk.message.thinking, end='', flush=True)
                    # accumulate the partial thinking 
                    thinking += chunk.message.thinking
                elif chunk.message.content:
                    if in_thinking:
                        in_thinking = False
                        print(Fore.GREEN + '\n\nAnswer:\n', end='', flush=True)
                    print(Fore.GREEN + chunk.message.content + Style.RESET_ALL, end='', flush=True)
                    # accumulate the partial content
                    content += chunk.message.content

            # append the completed assistant turn to the messages for the next request
            self.conversation.append({'role': 'assistant', 'content': content})
            print("\n")

            # once the HRI agent has resolved the command, hand the confirmed
            # intent to the planner for long-horizon decomposition.
            self.dispatch_to_planner(content)

            user_query = input("User: ")
            self.conversation.append({'role': 'user', 'content': user_query})

    def dispatch_to_planner(self, hri_output: str):
        """Parse an HRI turn and, if it reached EXECUTE with a confirmed intent,
        call the planner with that intent as the long-horizon task."""
        # The model may wrap its JSON in markdown fences or add stray prose, so
        # slice out the outermost {...} object before parsing.
        start, end = hri_output.find("{"), hri_output.rfind("}")
        if start == -1 or end == -1:
            print(Fore.RED + "[HRI] no JSON object found in output; skipping planner." + Style.RESET_ALL)
            return
        try:
            trace = json.loads(hri_output[start:end + 1]).get("trace", {})
        except (json.JSONDecodeError, AttributeError):
            print(Fore.RED + "[HRI] could not parse output as JSON; skipping planner." + Style.RESET_ALL)
            return

        confirmed_intent = trace.get("confirmed_intent")
        if trace.get("mode") == "EXECUTE" and confirmed_intent:
            self.planner_agent.plan(confirmed_intent, self.initial_frame_path)
            # after the (planned) execution, validate full task completeness
            # against the final frame of the episode.
            self.validator_agent.validate(confirmed_intent, self.final_frame_path)

