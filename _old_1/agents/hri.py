import json
import uuid
from pathlib import Path

from ollama import chat
from colorama import Fore, Style

from assurance.task_assurance import TaskAssurance
from agents.memory import Memory_Agent, Memory_Agent_Config
from agents.planner import Planner_Agent, Planner_Agent_Config
from agents.vision import prepare_vision_image
from agents.vlidator import Validator_Agent, Validator_Agent_Config
from dataset.episode import DatasetEpisode, DatasetEpisodeError
from memory.store import PreferenceStore
from agents.configs import HRI_Agent_Config



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
        self.task_assurance = TaskAssurance()
        self.last_task_result: dict | None = None
        self.pending_memory_confirmation: dict | None = None

        planner_config = Planner_Agent_Config()
        planner_config.resize_images = config.resize_images
        planner_config.image_width = config.image_width
        planner_config.image_height = config.image_height
        planner_config.image_jpeg_quality = config.image_jpeg_quality
        self.planner_agent = Planner_Agent(planner_config)

        validator_config = Validator_Agent_Config()
        validator_config.resize_images = config.resize_images
        validator_config.image_width = config.image_width
        validator_config.image_height = config.image_height
        validator_config.image_jpeg_quality = config.image_jpeg_quality
        self.validator_agent = Validator_Agent(validator_config)

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
            print(Fore.GREEN + '\nHRI:\n', end='', flush=True)
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

            if mode == "EXECUTE":
                while self.pending_memory_confirmation is not None:
                    print(Fore.BLUE + self.memory_confirmation_question() + Style.RESET_ALL)
                    self.answer_memory_confirmation(input("Memory preference: "))
                self.prompt_for_dataset_switch()
                user_query = input("User: ")
                self._append_user_turn(user_query, new_command=True)
            else:
                user_query = input("User: ")
                self._append_user_turn(user_query, new_command=mode == "REPORT")

    def _append_user_turn(self, user_query: str, new_command: bool) -> None:
        if new_command:
            self.conversation = [
                {
                    'role': 'system',
                    'content': self.system_prompt,
                }
            ]
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
            message['images'] = [
                prepare_vision_image(
                    self.initial_frame_path,
                    resize=self.config.resize_images,
                    width=self.config.image_width,
                    height=self.config.image_height,
                    jpeg_quality=self.config.image_jpeg_quality,
                )
            ]
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
        if mode == "EXECUTE" and not confirmed_intent:
            self.last_task_result = {
                "phase": "HRI",
                "outcome": "UNKNOWN",
                "proceed": False,
                "next_action": "REOBSERVE",
                "message": "The HRI output did not contain a confirmed task contract.",
                "failure": {
                    "stage": "GROUNDING",
                    "code": "MISSING_CONFIRMED_INTENT",
                    "memory_effect": "NONE",
                },
            }
            print(Fore.RED + "[HRI] EXECUTE requires a confirmed intent; skipping planner." + Style.RESET_ALL)
            return None
        if mode == "EXECUTE" and confirmed_intent:
            self.update_memory()
            try:
                planner_output = self.planner_agent.plan(confirmed_intent, self.initial_frame_path)
            except Exception as error:
                task_result = self.task_assurance.runtime_failure(
                    stage="PLANNING",
                    code="PLANNER_UNAVAILABLE",
                    message="The planner is unavailable, so no task was dispatched.",
                    next_action="REPLAN",
                    observed=str(error),
                )
            else:
                plan_result = self.task_assurance.assess_plan(planner_output)
                if plan_result.proceed:
                    # In the recorded-episode prototype, the final frame represents the
                    # state after the external VLA/robot attempt. The assurance layer only
                    # evaluates it; it does not claim to execute the physical action.
                    try:
                        validator_output = self.validator_agent.validate(
                            confirmed_intent, self.final_frame_path
                        )
                    except Exception as error:
                        task_result = self.task_assurance.runtime_failure(
                            stage="VALIDATION",
                            code="VALIDATOR_UNAVAILABLE",
                            message="The final state could not be validated, so success is unknown.",
                            next_action="REOBSERVE",
                            observed=str(error),
                        )
                    else:
                        task_result = self.task_assurance.assess_validation(validator_output)
                else:
                    task_result = plan_result
            self.last_task_result = task_result.to_dict()
            print(
                Fore.YELLOW
                + "[Task Assurance] "
                + json.dumps(self.last_task_result, indent=2)
                + Style.RESET_ALL
            )
        elif mode == "REPORT":
            failure_code = trace.get("failure_code") or "HRI_REPORTED_BLOCKER"
            if failure_code == "USER_CANCELLED":
                report_outcome, next_action = "CANCELLED", "NONE"
            elif failure_code == "UNSAFE_REQUEST":
                report_outcome, next_action = "ABORTED_SAFETY", "ABORT_SAFETY"
            elif failure_code == "UNSUPPORTED_TASK":
                report_outcome, next_action = "BLOCKED", "ABORT_UNSUPPORTED"
            else:
                report_outcome, next_action = "BLOCKED", "USER_ASSIST"
            self.last_task_result = {
                "phase": "HRI",
                "outcome": report_outcome,
                "proceed": False,
                "next_action": next_action,
                "message": trace.get("report_reason") or "The HRI agent reported that the task cannot proceed.",
                "failure": {
                    "stage": "GROUNDING",
                    "code": failure_code,
                    "memory_effect": "NONE",
                },
            }
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

    def update_memory(self) -> list[dict]:
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
            return []

        if applied:
            print(Fore.BLUE + "[Memory] " + json.dumps(applied) + Style.RESET_ALL)
        confirmation = next(
            (result for result in applied if result.get("action") == "CONFIRM_REQUIRED"),
            None,
        )
        if confirmation is not None:
            self.pending_memory_confirmation = confirmation
        return applied

    def memory_confirmation_question(self) -> str:
        """Return a dedicated future-memory question for the pending candidate."""
        pending = self.pending_memory_confirmation
        if pending is None:
            return ""
        value = json.dumps(pending.get("value"), ensure_ascii=False)
        task = str(pending.get("task_type", "this task")).replace("_", " ")
        if pending.get("reason") == "conflicts_with_durable":
            return f"You chose {value}, which differs from your saved default. Should it replace your default for {task}?"
        return f"You have independently chosen {value} more than once. Should I remember it as your default for {task}?"

    def answer_memory_confirmation(self, answer: str) -> dict | None:
        """Apply only an answer to the dedicated memory question as durable consent."""
        pending = self.pending_memory_confirmation
        if pending is None:
            return None
        normalised = " ".join(answer.lower().strip().rstrip(".!?").split())
        yes_answers = {"yes", "yeah", "yep", "sure", "ok", "okay", "please do", "remember it"}
        no_answers = {"no", "nope", "do not", "don't", "not now", "this time only"}
        if normalised in yes_answers:
            result = self.memory_store.confirm_candidate(
                pending["preference_id"],
                user_id=self.config.user_id,
                session_id=self.session_id,
                quote=answer,
            )
        elif normalised in no_answers:
            result = self.memory_store.decline_candidate(
                pending["preference_id"],
                user_id=self.config.user_id,
                session_id=self.session_id,
                quote=answer,
            )
        else:
            print(Fore.BLUE + "Please answer yes or no about remembering the preference." + Style.RESET_ALL)
            return None
        self.pending_memory_confirmation = None
        print(Fore.BLUE + "[Memory] " + json.dumps(result) + Style.RESET_ALL)
        return result

