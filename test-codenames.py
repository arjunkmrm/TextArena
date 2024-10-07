from textarena.envs.multi_player.codenames import CodenamesEnv
import time, os
from openai import OpenAI

from textarena.wrappers import (
    PrettyRenderWrapper,
    LLMObservationWrapper,
    ClipWordsActionWrapper
)

import textarena

textarena.pprint_registry_detailed()
class GPTAgent:
    def __init__(self, model_name: str):
        """
        Initialize the GPTAgent with the specified OpenRouter model.
        
        Args:
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-4").
        """
        self.model_name = model_name
        self.agent_identifier = model_name
        # Load the OpenAI API key from environment variable
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        
    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenRouter model and return the action.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The response generated by the model.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": observation}
                ],
                temperature=0.7,
            )
            # Extract the assistant's reply
            action = response.choices[0].message.content.strip()
            return action
        except Exception as e:
            return f"An error occurred: {e}"

agent_0 = GPTAgent(
    model_name="gpt-4o-mini"
)

agent_1 = GPTAgent(
    model_name="gpt-4o-mini"
)

env = textarena.make("Codenames-v0-basic")

env = LLMObservationWrapper(env=env)

env = PrettyRenderWrapper(env=env, max_log_lines=20)

observations, info = env.reset()
print(info.get("words_and_roles"))

done = False
while not done:
    if env.game_state["current_role"] == "spymaster" and env.game_state["current_team"] == 0:
        action = agent_0(
            observations[0]
        )
        print('****************************************************************************************')
        print("Agent 0 Observations:", observations[0])
        print('****************************************************************************************')
        print("Agent 0 action:", action)
        print('****************************************************************************************')
        observations, reward, truncated, terminated, info = env.step(0, action)
        env.render()
        time.sleep(1)

    elif env.game_state["current_role"] == "operative" and env.game_state["current_team"] == 0:
        action = agent_1(
            observations[1]
        )
        print('****************************************************************************************')
        print("Agent 1 Observations:", observations[1])
        print('****************************************************************************************')
        print("Agent 1 action:", action)
        print('****************************************************************************************')
        observations, reward, truncated, terminated, info = env.step(1, action)
        env.render()
        time.sleep(1)

    elif env.game_state["current_role"] == "spymaster" and env.game_state["current_team"] == 1:
        action = agent_1(
            observations[2]
        )
        print('****************************************************************************************')
        print("Agent 2 Observations:", observations[2])
        print('****************************************************************************************')
        print("Agent 2 action:", action)
        print('****************************************************************************************')
        observations, reward, truncated, terminated, info = env.step(2, action)
        env.render()
        time.sleep(1)

    elif env.game_state["current_role"] == "operative" and env.game_state["current_team"] == 1:
        action = agent_1(
            observations[3]
        )
        print('****************************************************************************************')
        print("Agent 3 Observations:", observations[3])
        print('****************************************************************************************')
        print("Agent 3 action:", action)
        print('****************************************************************************************')
        observations, reward, truncated, terminated, info = env.step(3, action)
        env.render()
        time.sleep(1)

    done = truncated or terminated