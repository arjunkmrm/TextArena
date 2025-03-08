"""
uv run --with nltk --with anthropic run_online_simple.py
"""

import textarena as ta
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint
from rich import print as rprint

model_name = "Gabriel - Test"
model_description = "Test model for Gabriel"
email = "cyzgab@gmail.com"

SYSTEM_PROMPT = """
You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.

For example:
- if the game is SpellingBee-v0, the observation will be a list of words, and the action (i.e., your REPLY) will be a word (e.g., `[hello]`)
- if the game is SimpleNegotiation-v0, the observation will be a list of the current state of the negotiation, and the action will be a message to send to the other player (e.g., `[Offer: <your resources> -> <their resources>]`).
- if the game is Poker-v0, the observation will be a list of the current state of the game, and the action will be a move to make (e.g., `[Bet 100]`).

Reply in valid JSON with the following keys:
- `thinking`: use this as a scratchpad to think about the game.
- `action`: the action to take.
"""

# Initialize agent
agent = ta.agents.AnthropicAgent(model_name="claude-3-7-sonnet-20250219", temperature=1, verbose=True, system_prompt=SYSTEM_PROMPT, thinking=True) 


env = ta.make_online(
    env_id=["SpellingBee-v0", "SimpleNegotiation-v0", "Poker-v0"], 
    model_name=model_name,
    model_description=model_description,
    email=email
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False
console = Console()
while not done:
    console.rule("[bold cyan]New Turn", style="cyan")
    player_id, observation = env.get_observation()
    console.print(f"[bold green]Player ID:[/bold green] {player_id}")
    
    # Use repr() instead of str() to properly display brackets in the observation
    console.print(Panel.fit(
        repr(observation),
        title="[bold blue]OBSERVATION[/bold blue]",
        border_style="blue"
    ))
    
    # Convert observation to string for the agent
    observation_str = str(observation)
    action = agent(observation_str)
    
    console.print(Panel.fit(
        repr(action),
        title="[bold magenta]ACTION[/bold magenta]",
        border_style="magenta"
    ))
    
    done, info = env.step(action=action)
    
    console.print(f"[bold yellow]DONE:[/bold yellow] {done}")
    
    console.print(Panel.fit(
        repr(info) if info is not None else "None",
        title="[bold red]INFO[/bold red]",
        border_style="red"
    ))


rewards = env.close()