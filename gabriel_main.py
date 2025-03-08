"""
uv run --with nltk --with anthropic run_online_simple.py
"""

import textarena as ta
from rich.console import Console
from rich.panel import Panel

model_name = "Gabriel - Test"
model_description = "Test model for Gabriel"
email = "cyzgab@gmail.com"

GAMES = {
    "SpellingBee-v0": False,
    "SimpleNegotiation-v0": True,
    "Poker-v0": False,
}

GAMES_TO_RUN = [k for k, v in GAMES.items() if v]

SYSTEM_PROMPT = """
You are an elite, competitive game player who has a PhD in Game Theory. Your goal is to analyze the game instructions, process the observation, and make the optimal move.

1. **Read the Instructions**: Always carefully read the game instructions before taking any action. The game rules may be more complex than what is evident in the observation.

2. **Response Format**: Your reply must be in valid JSON with exactly two keys:
   - `thinking`: A scratchpad where you document your reasoning process.
   - `anticipated_moves`: A list of moves that you anticipate the other player will make.
   - `thinking_about_anticipated_moves`: A scratchpad where you document your reasoning about the anticipated moves, and how you will respond to them.
   - `action`: Your chosen move or response based on the game state.

3. **Game Examples**:
   - **SpellingBee-v0**: 
     - **Observation**: A list of words.
     - **Action**: Choose one word (e.g., `"hello"`).
     - **Example Output**: `{"thinking": "Reviewing available words.", "action": "hello"}`
     
   - **SimpleNegotiation-v0**: 
     - **Observation**: The current state of the negotiation.
     - **Action**: Send a negotiation message (e.g., `"Offer: <your resources> -> <their resources>"`).
     - **Example Output**: `{"thinking": "Evaluating negotiation state.", "action": "Offer: 50 -> 30"}`
     
   - **Poker-v0**: 
     - **Observation**: The current state of the poker game.
     - **Action**: Make a move (e.g., `"Bet 100"`).
     - **Example Output**: `{"thinking": "Assessing hand strength.", "action": "Bet 100"}`

4. Remember to use the tools provided to you to make your move, if they are relevant to the game.
     
5. **Adaptability**: Remember that the actual game may include complexities not immediately apparent from the observation. Always refer to the detailed game instructions when making your decision.
""".strip()

# Initialize agent
# agent = ta.agents.AnthropicAgent(model_name="claude-3-7-sonnet-20250219", temperature=1, system_prompt=SYSTEM_PROMPT, json_prefill=True) 
agent = ta.agents.OpenAIAgent(model_name="o3-mini", temperature=1, system_prompt=SYSTEM_PROMPT, response_format={"type": "json_object"}, store=True, reasoning_effort="medium")
# agent = ta.agents.GeminiAgent(model_name="gemini-2.0-pro-exp-02-05", system_prompt=SYSTEM_PROMPT)

env = ta.make_online(
    env_id=GAMES_TO_RUN, 
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
    
    if info is not None:
        console.print(Panel.fit(
            repr(info) if info is not None else "None",
            title="[bold red]INFO[/bold red]",
            border_style="red"
        ))


rewards = env.close()