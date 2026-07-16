import json
import uuid
from pathlib import Path

from ollama import chat
from colorama import Fore, Style

from agents.memory import Memory_Agent, Memory_Agent_Config
from agents.planner import Planner_Agent, Planner_Agent_Config
from agents.vlidator import Validator_Agent, Validator_Agent_Config
from dataset.episode import DatasetEpisode, DatasetEpisodeError
from memory.store import PreferenceStore



class HRI_Agent_Config:
    host: str = "ollama"
    base_url: str = ""
    model: str = "gemma4:31b-cloud"

    workspace_root: str = str(Path(__file__).resolve().parents[1])
    system_prompt_path: str = "/home/yuxin/workspace/RoboPref/prompt/hri/prompt-v1.md"
    dataset_path: str = str(Path(workspace_root) / "dataset" / "v3")
    memory_store_path: str = str(Path(__file__).resolve().parents[1] / "memory" / "preferences.json")
    user_id: str = "default"


class HRI_Agent:
    def __init__(self, config: HRI_Agent_Config):
        self.config = config
        self.host = config.host
        self.base_url = config.base_url
        self.model = config.model
        self.workspace_root = Path(config.workspace_root).resolve()
        self.dataset_episode = DatasetEpisode.from_path(config.dataset_path, self.workspace_root)
        with open(config.system_prompt_path, "r") as f:
            self.system_prompt = f.read()
        self.conversation = []
        self.current_interaction = []
        self.relevant_memories = []
        self.session_id = uuid.uuid4().hex
        self.memory_store = PreferenceStore(config.memory_store_path)
        self.memory_agent = Memory_Agent(Memory_Agent_Config())
        self.planner_agent = Planner_Agent(Planner_Agent_Config())
        self.validator_agent = Validator_Agent(Validator_Agent_Config())

    @property
    def initial_frame_path(self) -> str:
        return str(self.dataset_episode.initial_frame)

    @property
    def final_frame_path(self) -> str:
        return str(self.dataset_episode.final_frame)

    def get_response(self):
        print("\n--- Start of Conversation ---\n")
        user_query = input("User: ")
        if not user_query:
            user_query = "Stacked vertically the blocks from the right mat on the left mat. RGB sequence from bottom to top."
            print(f"Using default user query: {user_query}")
        self.conversation.append({
            'role': 'system',
            'content': self.system_prompt
        })
        self._append_user_turn(user_query, new_command=True)

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
            self.current_interaction.append({'role': 'assistant', 'content': content})
            print("\n")

            # once the HRI agent has resolved the command, hand the confirmed
            # intent to the planner for long-horizon decomposition.
            mode = self.dispatch_to_planner(content)

            user_query = input("User: ")
            self._append_user_turn(user_query, new_command=mode in {"EXECUTE", "REPORT"})

    def _append_user_turn(self, user_query: str, new_command: bool) -> None:
        if new_command:
            self.current_interaction = []
            self.relevant_memories = self.memory_store.retrieve(user_query, user_id=self.config.user_id)

        self.current_interaction.append({'role': 'user', 'content': user_query})
        payload = {
            "user_message": user_query,
            "memory": self.relevant_memories,
        }
        message = {
            'role': 'user',
            'content': json.dumps(payload),
        }
        if new_command:
            message['images'] = [self.initial_frame_path]
        self.conversation.append(message)

    def dispatch_to_planner(self, hri_output: str) -> str | None:
        """Parse an HRI turn and, if it reached EXECUTE with a confirmed intent,
        call the planner with that intent as the long-horizon task."""
        # The model may wrap its JSON in markdown fences or add stray prose, so
        # slice out the outermost {...} object before parsing.
        start, end = hri_output.find("{"), hri_output.rfind("}")
        if start == -1 or end == -1:
            print(Fore.RED + "[HRI] no JSON object found in output; skipping planner." + Style.RESET_ALL)
            return None
        try:
            trace = json.loads(hri_output[start:end + 1]).get("trace", {})
        except (json.JSONDecodeError, AttributeError):
            print(Fore.RED + "[HRI] could not parse output as JSON; skipping planner." + Style.RESET_ALL)
            return None

        mode = trace.get("mode")
        confirmed_intent = trace.get("confirmed_intent")
        if mode == "EXECUTE" and confirmed_intent:
            self.update_memory()
            self.planner_agent.plan(confirmed_intent, self.initial_frame_path)
            # after the (planned) execution, validate full task completeness
            # against the final frame of the episode.
            self.validator_agent.validate(confirmed_intent, self.final_frame_path)
            self.prompt_for_dataset_switch()
        return mode

    def prompt_for_dataset_switch(self) -> bool:
        """Offer an episode switch for the next command after validation completes."""
        current_path = self.dataset_episode.display_path(self.workspace_root)
        selected_path = input(
            f"Dataset for next task [{current_path}] (Enter to keep): "
        ).strip()
        if not selected_path:
            print(Fore.CYAN + f"Keeping dataset: {current_path}" + Style.RESET_ALL)
            return False

        try:
            episode = DatasetEpisode.from_path(selected_path, self.workspace_root)
        except DatasetEpisodeError as error:
            print(Fore.RED + f"[Dataset] {error}" + Style.RESET_ALL)
            print(Fore.CYAN + f"Keeping dataset: {current_path}" + Style.RESET_ALL)
            return False

        self.dataset_episode = episode
        print(Fore.CYAN + f"Switched dataset: {episode.display_path(self.workspace_root)}" + Style.RESET_ALL)
        print(Fore.CYAN + f"Initial frame: {self._display_path(episode.initial_frame)}" + Style.RESET_ALL)
        print(Fore.CYAN + f"Final frame: {self._display_path(episode.final_frame)}" + Style.RESET_ALL)
        return True

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def update_memory(self) -> None:
        """Curate the resolved interaction without allowing memory failure to block execution."""
        try:
            operations = self.memory_agent.propose_updates(
                self.current_interaction,
                self.relevant_memories,
            )
            applied = self.memory_store.apply_operations(
                operations,
                user_id=self.config.user_id,
                session_id=self.session_id,
            )
        except Exception as error:
            print(Fore.RED + f"[Memory] update failed; continuing without persistence: {error}" + Style.RESET_ALL)
            return

        if applied:
            print(Fore.BLUE + "[Memory] " + json.dumps(applied) + Style.RESET_ALL)

