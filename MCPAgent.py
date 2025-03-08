import os
import json
import textarena as ta
import smithery
import mcp
from textarena.agents.basic_agents import AsyncAnthropicAgent

class MCPAgent(AsyncAnthropicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/e2b/ws", {"e2bApiKey": os.environ["E2B_API_KEY"]}
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
                    
                    # Add counter to limit tool calls
                    tool_call_count = 0

                    # Loop to handle multiple tool calls in a conversation
                    while is_tool_call_pending:
                        # Check if we've reached the maximum number of tool calls
                        if tool_call_count >= 3:
                            print("Reached maximum number of tool calls (3)")
                            # Add a message to final response if there's none yet
                            if not final_response_text:
                                final_response_text = "Reached maximum number of tool calls (3). Providing partial results."
                            is_tool_call_pending = False
                            break
                            
                        response = await self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system=self.system_prompt,
                            messages=messages,
                            tools=tools,
                            tool_choice={"type": "any"}
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

                                # Increment tool call counter
                                tool_call_count += 1
                                print(f"Tool call {tool_call_count}/3: {tool_name}")

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
                                # print(f"Tool result: {result_str}")

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

                        # After processing all content blocks for this response and handling tool calls
                        # Only add Assistant Pre-Fill if there are more tool calls to process
                        if is_tool_call_pending:
                            # Add Assistant Pre-Fill after tool calls are processed
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": "```json\n{\"thinking\"}:"}],
                                }
                            )

                        # If no tool calls were made, we use the text response
                        if not is_tool_call_pending and not final_response_text:
                            # Safer access to text content
                            for block in response.content:
                                if hasattr(block, "text"):
                                    final_response_text = block.text
                                    break
                            
                            final_response_text = "{\"thinking\"}" + final_response_text
                            final_response_text = json.loads(final_response_text)["action"]

                except Exception as e:

                    print(f"Error: {e}")
                    raise e

            return final_response_text.strip()