
from textarena.core import Agent
import textarena as ta
import asyncio
from typing import Optional

GAMES = {
    "SpellingBee-v0": True,
    "SimpleNegotiation-v0": False,
    "Poker-v0": False,
}

GAMES_TO_RUN = [k for k, v in GAMES.items() if v]

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
   Always use the available tools if they can help inform your decision.

6. **Flexibility:**  
   Games may include hidden complexities. Always refer back to the full instructions when deciding your move.

Begin directly with your JSON output code block. Don't give any explanation or commentary.
""".strip()

class AsyncAnthropicAgent(Agent):
    """Agent class using the Anthropic Claude API to generate responses asynchronously."""
    def __init__(self, model_name: str, system_prompt: Optional[str] = STANDARD_GAME_PROMPT, max_tokens: int = 1000, temperature: float = 0.9, verbose: bool = False):
        """
        Initialize the Anthropic agent.

        Args:
            model_name (str): The name of the Claude model (e.g., "claude-3-5-sonnet-20241022").
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for randomness in response generation.
            verbose (bool): If True, additional debug info will be printed.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required for AsyncAnthropicAgent. "
                "Install it with: pip install anthropic"
            )
            
        self.client = anthropic.AsyncAnthropic()
    
    async def _make_request(self, observation: str) -> str:
        """Make a single API request to Anthropic and return the generated message."""
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
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
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
            observation (str): The input string to process.
        
        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return await self._retry_request(observation)
    
import textarena as ta
import smithery
import mcp
import os
import json


class MCPAgent(AsyncAnthropicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/@kwen1510/nltk-map/ws"
        )

    async def _make_request(self, observation: str) -> str:
        """Make a single API request to Anthropic and return the generated message."""
        async with smithery.websocket_client(self.url) as streams:
            async with mcp.client.session.ClientSession(*streams) as session:

                try:
                    tools_result = await session.list_tools()
                    tools = tools_result.model_dump()["tools"]

                    tools = [
                        {"input_schema": tool.pop("inputSchema"), **tool}
                        for tool in tools
                        if "inputSchema" in tool
                    ]

                    print("Available tools:", tools)

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

                                    # Convert tool result to string format for Anthropic
                                    # The content must be a string, not an object
                                    tool_result_dict = tool_result.model_dump()
                                except Exception as e:
                                    if "MCP error" in str(e):
                                        tool_result_dict = {"error": str(e)}

                                result_str = json.dumps(tool_result_dict)
                                print(f"Tool result: {result_str}")

                                # Add tool call and result to messages
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [content_block.model_dump()],
                                    }
                                )

                                # Add tool response to messages - content must be a string
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_id,
                                                "content": result_str,  # Now it's a string
                                            }
                                        ],
                                    }
                                )
                            elif content_block.type == "text":
                                # Accumulate text responses
                                final_response_text += content_block.text

                        # If no tool calls were made, we use the text response
                        if not is_tool_call_pending and not final_response_text:
                            final_response_text = response.content[0].text

                except Exception as e:

                    print(f"Error: {e}")
                    raise e
                
            final_text = final_response_text.strip()
            try:
                final_text = final_text.split("```json")[1]
            except:
                try:
                    final_text = final_text.split("```")[1]
                except:
                    pass
            final_text = final_text.replace("```json", "").replace("```", "")
            print(f"Final text: \n {final_text}")
            final_text = json.loads(final_text)["action"]


            return final_text

import textarena as ta

# Initialize agents
agents = MCPAgent(model_name="claude-3-7-sonnet-20250219")

# Initialize environment from subset and wrap it
env = ta.make_online(
    env_id=GAMES_TO_RUN, 
    model_name="Test 123456789",
    model_description="Test 123456789",
    email="Test 123456789"
)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False

while not done:
    player_id, observation = env.get_observation()
    print(f"##################### {player_id} #####################")
    print(f"Observation: {observation}")
    action = asyncio.get_event_loop().run_until_complete(agents(observation))
    print(f"Action: {action}")
    done, info = env.step(action=action)
    print(f"Done: {done}")
    print(f"Info: {info}")
    print("step complete")
    
rewards = env.close()
print(rewards)