import os
import time

import textarena
from textarena.utils import batch_open_router_generate, DEFAULT_GEN_KWARGS
from textarena.envs.two_player.iterated_prisoners_dilemma import (
    IteratedPrisonersDilemma,
)
from textarena.wrappers import (
    ClipWordsActionWrapper,
    LLMObservationWrapper,
    PrettyRenderWrapper,
)

textarena.pprint_registry_detailed()


class GPTAgent:
    def __init__(self, model_name: str):
        """
        Initialize the GPTAgent with the specified OpenAI model.

        Args:
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-4").
        """
        self.model_name = model_name
        self.agent_identifier = model_name
        self.auth_token = os.getenv("OPEN_ROUTER_TOKEN")

    def __call__(self, str_rep: list[str], gen_kwargs=DEFAULT_GEN_KWARGS) -> list[str]:
        """
        Process the observation using the OpenAI model and return the action.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The response generated by the model.
        """
        responses = batch_open_router_generate(
            texts=observations,
            model_string=self.model_name,
            message_history=[
                {
                    "role": "system",
                    "content": "You are a helpful game-playing assistant.",
                }
            ],
            **gen_kwargs,
        )
        return responses


# build agents
agent_0 = GPTAgent(model_name="gpt-4o-mini")

agent_1 = GPTAgent(model_name="gpt-3.5-turbo")

# env = DontSayItEnv(hardcore=True)
# env = textarena.make("DontSayIt-v0-hardcore")
textarena.register(
    "IteratedPrisonersDilemma-v0",
    lambda: IteratedPrisonersDilemma(chat_turns_per_round=1, max_turns=30),
)
env = textarena.make("IteratedPrisonersDilemma-v0")

# wrap for LLM use
env = LLMObservationWrapper(env=env)

# env = ClipWordsActionWrapper(env, max_num_words=150)

# # wrap env
# env = PrettyRenderWrapper(
#     env=env,
#     agent_identifiers={0: agent_0.agent_identifier, 1: agent_1.agent_identifier},
# )


observations, info = env.reset()
# input(env.game_state)

done = False
while not done:
    for player_id, agent in enumerate([agent_0, agent_1]):

        # get the agent prompt
        action = agent([observations[player_id]])[0]
        print(action)

        observations, reward, truncated, terminated, info = env.step(player_id, action)
        env.render()
        print(info)
        time.sleep(1)

        done = truncated or terminated

        if done:
            break

for l in env.game_state["logs"]:
    print(l, end="\n\n")
