from ollama import chat
from colorama import Fore, Style

from agents.vision import prepare_vision_image
from agents.configs import Planner_Agent_Config


class Planner_Agent:
    def __init__(self, config: Planner_Agent_Config):
        self.config = config
        self.host = config.host
        self.base_url = config.base_url
        self.model = config.model
        with open(config.system_prompt_path, "r") as f:
            self.system_prompt = f.read()

    def plan(self, confirmed_intent: str, image_path: str):
        """Decompose a confirmed long-horizon intent into short-horizon subtasks."""
        image = prepare_vision_image(
            image_path,
            resize=self.config.resize_images,
            width=self.config.image_width,
            height=self.config.image_height,
            jpeg_quality=self.config.image_jpeg_quality,
        )
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': f"TASK TO PLAN: {confirmed_intent}",
                'images': [image]
            }
        ]

        stream = chat(
            model=self.model,
            messages=messages,
            think=False,
            stream=True,
            format="json",
            options={
                "temperature": 0.0
            }
        )

        print(Fore.CYAN + '\nPlanner:\n', end='', flush=True)
        content = ''
        for chunk in stream:
            if chunk.message.content:
                print(Fore.CYAN + chunk.message.content + Style.RESET_ALL, end='', flush=True)
                content += chunk.message.content
        print("\n")

        return content
