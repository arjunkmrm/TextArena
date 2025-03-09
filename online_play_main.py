"""
TextArena Game Agent using Model Context Protocol (MCP)

This module implements a game-playing agent using Anthropic's Claude API
with MCP (Model Context Protocol) for enhanced decision making in
TextArena games.
"""

from textarena.core import Agent
import textarena as ta
import asyncio
import json
from typing import Optional
import smithery
import mcp

# Dictionary of available games, with True for games to be played
GAMES = {
    "SpellingBee-v0": False,
    "SimpleNegotiation-v0": False,
    "Poker-v0": False,
    "Nim-v0": True,
    "TruthAndDeception-v0": False,
    "Snake-v0": False,
}

# Filter active games
GAMES_TO_RUN = [k for k, v in GAMES.items() if v]

# MCP (Model Context Protocol) URL identifier
# MCP = "@arjunkmrm/textarena-mcp"
MCP = "@watsonchua/poker_win_calculator"

# Prompt template for the agent's system instructions
STANDARD_GAME_PROMPT = """
You are an elite competitive game player with a PhD in Game Theory. Your mission is to analyze the game instructions, assess the observation, and choose the optimal move to win as fast as possible.

**Instructions:**

1. **Read the Rules:**  
   Carefully review all game instructions. The observation might not reveal every detail.

2. **Output Format:**  
   Respond in valid JSON with exactly four keys:
   - `"thinking"`: Your internal reasoning process.
   - `"anticipated_moves"`: A list of moves you expect the opponent to make.
   - `"thinking_about_anticipated_moves"`: Your reasoning on how to counter their moves.
   - `"action"`: The move you choose based on the game state.

3. **Examples:**
   - *SpellingBee-v0*  
     - **Observation:** A list of words.  
     - **Action:** Choose one word (e.g., `"hello"`).  
     - **Output Example:**  
       `{"thinking": "Reviewing available words.", "anticipated_moves": [], "thinking_about_anticipated_moves": "", "action": "hello"}`
   
   - *SimpleNegotiation-v0*  
     - **Observation:** The current negotiation state.  
     - **Action:** Send a negotiation message (e.g., `"Offer: 50 -> 30"`).  
     - **Output Example:**  
       `{"thinking": "Evaluating negotiation state.", "anticipated_moves": [], "thinking_about_anticipated_moves": "", "action": "Offer: 50 -> 30"}`
   
   - *Poker-v0*  
     - **Observation:** The current state of the poker game.  
     - **Action:** Make a move (e.g., `"Bet 100"`).  
     - **Output Example:**  
       `{"thinking": "Assessing hand strength.", "anticipated_moves": [], "thinking_about_anticipated_moves": "", "action": "Bet 100"}`

4. **Strategy Guidelines:**
   - For *SpellingBee-v0*: Use the tool to select the longest possible word in your first turn.
   - For *Poker-v0*: Follow the guidance provided by the tool.

5. **Tool Use:**  
   Only use the respective tool for spelling bee and poker.
   JUST TAKE THE ANSWER FROM THE TOOL AS THE TRUTH

6. **Flexibility:**  
   Games may include hidden complexities. Always refer back to the full instructions when deciding your move.

Begin directly with your JSON output code block. Don't give any explanation or commentary.
""".strip()


class AsyncAnthropicAgent(Agent):
    """
    Agent class using the Anthropic Claude API to generate responses asynchronously.
    Provides a framework for making requests to Anthropic's API with retries.
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        max_tokens: int = 1000,
        temperature: float = 0,
        verbose: bool = False
    ):
        """
        Initialize the Anthropic agent.

        Args:
            model_name: The name of the Claude model (e.g., "claude-3-5-sonnet-20241022")
            system_prompt: The system prompt to use
            max_tokens: The maximum number of tokens to generate
            temperature: The temperature for randomness in response generation
            verbose: If True, additional debug info will be printed
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Dynamically import anthropic to handle the dependency gracefully
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required for AsyncAnthropicAgent. "
                "Install it with: pip install anthropic"
            )
            
        self.client = anthropic.AsyncAnthropic()
    
    async def _make_request(self, observation: str) -> str:
        """
        Make a single API request to Anthropic and return the generated message.
        
        Args:
            observation: The input string to process
            
        Returns:
            The generated response text
        """
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": observation}]}
            ]
        )
        
        return response.content[0].text.strip()
    
    async def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation: The input to process
            retries: The number of attempts to try
            delay: Seconds to wait between attempts

        Returns:
            The generated response text
            
        Raises:
            Exception: The last exception caught if all retries fail
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = await self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    await asyncio.sleep(delay)
        raise last_exception
    
    async def __call__(self, observation: str) -> str:
        """
        Process the observation using the Anthropic API and return the generated response.
        
        Args:
            observation: The input string to process
        
        Returns:
            The generated response
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return await self._retry_request(observation)


class MCPAgent(AsyncAnthropicAgent):
    """
    Agent that extends AsyncAnthropicAgent to use Model Context Protocol (MCP)
    functionality for enhanced decision making.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the MCP agent with the same parameters as AsyncAnthropicAgent"""
        super().__init__(*args, **kwargs)

        # Initialize MCP connection URL
        self.url = smithery.create_smithery_url(f"wss://server.smithery.ai/{MCP}/ws")

    async def _make_request(self, observation: str) -> str:
        """
        Make a request to the Anthropic API with tool use capability via MCP.
        
        Args:
            observation: The input to process
            
        Returns:
            The action extracted from the LLM response
        """
        async with smithery.websocket_client(self.url) as streams:
            async with mcp.client.session.ClientSession(*streams) as session:
                try:
                    # Get available tools from MCP
                    tools_result = await session.list_tools()
                    tools = tools_result.model_dump()["tools"]

                    # Format tools for Anthropic API
                    tools = [
                        {"input_schema": tool.pop("inputSchema"), **tool}
                        for tool in tools
                        if "inputSchema" in tool
                    ]

                    print("Available tools:", tools)

                    # Initialize variables for conversation flow
                    final_response_text = ""
                    is_tool_call_pending = True
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": observation}],
                        }
                    ]

                    # Loop to handle multiple tool calls in a conversation
                    while is_tool_call_pending:
                        response = await self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system=self.system_prompt,
                            messages=messages,
                            tools=tools,
                        )

                        print("Response:", response)

                        # Check if there's a tool_use in the response
                        is_tool_call_pending = False
                        for content_block in response.content:
                            if content_block.type == "tool_use":
                                is_tool_call_pending = True

                                # Extract tool call details
                                tool_name = content_block.name
                                tool_input = content_block.input
                                tool_id = content_block.id

                                print(f"Tool called: {tool_name}")
                                print(f"Tool input: {json.dumps(tool_input, indent=2)}")

                                # Execute the tool using MCP session
                                try:
                                    tool_result = await session.call_tool(
                                        tool_name, tool_input
                                    )
                                    tool_result_dict = tool_result.model_dump()
                                except Exception as e:
                                    if "MCP error" in str(e):
                                        tool_result_dict = {"error": str(e)}

                                # Convert tool result to string for Anthropic API
                                result_str = json.dumps(tool_result_dict)
                                print(f"Tool result: {result_str}")

                                # Add tool call to messages
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [content_block.model_dump()],
                                    }
                                )

                                # Add tool response to messages
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_id,
                                                "content": result_str,
                                            }
                                        ],
                                    }
                                )
                            elif content_block.type == "text":
                                # Accumulate text responses
                                final_response_text += content_block.text

                        # If no tool calls were made, use the text response
                        if not is_tool_call_pending and not final_response_text:
                            final_response_text = response.content[0].text

                except Exception as e:
                    print(f"Error: {e}")
                    raise e
                
            # Extract and clean the response
            final_text = final_response_text.strip()
            
            # Try to extract the JSON part from the response
            try:
                # First attempt: look for {"thinking": pattern
                final_text = final_text.split("{\"thinking\":")[1]
                final_text = "{\"thinking\":" + final_text
            except:
                try:
                    # Second attempt: look for code blocks
                    final_text = final_text.split("```")[1]
                    final_text = "```" + final_text
                except:
                    pass
                    
            # Remove JSON and code block markers
            final_text = final_text.replace("```json", "").replace("```", "")
            print(f"Final text: \n {final_text}")
            
            # Parse JSON and extract just the action field
            final_text = json.loads(final_text)["action"]

            return final_text


def main():
    """Main function to run the TextArena environment with the MCP agent"""
    # Initialize the MCP agent with the latest Claude model
    agent = MCPAgent(model_name="claude-3-7-sonnet-20250219")

    # Initialize TextArena environment
    env = ta.make_online(
        env_id=GAMES_TO_RUN,
        model_name="sonnet-latest",
        model_description="sonnet-latest",
        email="sonnet-latest"
    )
    
    # Wrap environment for LLM compatibility
    env = ta.wrappers.LLMObservationWrapper(env=env)

    # Reset the environment
    env.reset(num_players=1)

    done = False
    # Main game loop
    while not done:
        # Get the current player and observation
        player_id, observation = env.get_observation()
        print(f"##################### {player_id} #####################")
        print(f"Observation: {observation}")
        
        # Get action from the agent
        action = asyncio.get_event_loop().run_until_complete(agent(observation))
        print(f"Action: {action}")
        
        # Execute the action in the environment
        done, info = env.step(action=action)
        print(f"Done: {done}")
        print(f"Info: {info}")
        print("step complete")

    # Close environment and get final rewards
    rewards = env.close()
    print(rewards)


if __name__ == "__main__":
    main()