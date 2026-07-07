from ollama import chat
from colorama import Fore, Back, Style



class HRI_Agent_Config:
    host: str = "ollama"
    base_url: str = ""
    model: str = "qwen3.5:397b-cloud"

    system_prompt_path: str = "/home/yuxin/workspace/RoboPref/prompt/hri/prompt-v1.md"


class HRI_Agent:
    def __init__(self, config: HRI_Agent_Config):
        self.config = config
        self.host = config.host
        self.base_url = config.base_url
        self.model = config.model
        with open(config.system_prompt_path, "r") as f:
            self.system_prompt = f.read()
        self.conversation = []

    def get_response(self):
        print("\n--- Start of Conversation ---\n")
        user_query = input("User: ")
        self.conversation.extend([
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user', 
                'content': user_query,
                'images': ["/home/yuxin/workspace/RoboPref/dataset/v1/1.jpg"]
            }
        ])

        


        while True:
            stream = chat(
                model=self.model,
                messages=self.conversation,
                think=False,
                stream=True,
                format = "json"
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

                # append the accumulated fields to the messages for the next request
                self.conversation.extend([{ 'role': 'assistant', 'content': content }])
            print("\n")
            user_query = input("User: ")
            self.conversation.append({'role': 'user', 'content': user_query})

