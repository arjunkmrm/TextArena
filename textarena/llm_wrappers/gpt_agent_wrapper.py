import os, openai 



class GPTAgent:
    def __init__(self, unique_identifier, api_key, verbose=False, max_tokens=1000, model_name="gpt-4o-mini"):
        """
        Initialize the GPT-4 agent.

        Args:
            unique_identifier (int): A unique identifier for the agent.
            api_key (str): Your OpenAI API key.
            max_tokens (int): Maximum number of tokens to generate.
            model_name (str): The model to use, e.g., 'gpt-4'.
        """
        self.unique_identifier = unique_identifier
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.history = []  # Stores tuples of (prompt, response)
        self.main_prompt = ""
        self.verbose = verbose 

        openai.api_key = self.api_key

    def reset(self, game_prompt):
        """
        Reset the agent with a new main prompt.

        Args:
            game_prompt (str): The main prompt or instructions for the player.
        """
        self.main_prompt = game_prompt
        self.history = []  # Clear history for a new game

    def get_action(self, observation, valid_actions=None):
        """
        Use the OpenAI GPT-4 API to generate an action.

        Args:
            state (str): The current state of the game specific to the player.
            valid_actions (list, optional): A list of valid actions.

        Returns:
            action (str): The generated action from the model.
            prompt (str): The full prompt sent to the model.
        """

        #print(f"\n[Player - GPT-4 API Agent]")
        #print(f"{prompt}")

        # Construct the messages for the chat completion
        messages = [{"role": "system", "content": self.main_prompt}]
        for h_state, h_action in self.history:
            messages.append({"role": "user", "content": h_state})
            messages.append({"role": "assistant", "content": h_action})
        # append valid actions
        if valid_actions:
            messages.append({"role": "user", "content": f"Valid actions: {', '.join(valid_actions)}\n"})
        if observation is not None and len(observation) > 0:
            messages.append({"role": "user", "content": observation})
        

        if self.verbose:
            prompt = ""
            for message in messages:
                prompt += f"\n{message['role']}: {message['content']}"
            print(prompt)


        # Call the GPT-4 API to generate a response
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Extract the generated action from the API response
        action = response['choices'][0]['message']['content'].strip()

        # Add to history
        self.history.append((
            observation,
            f"{action}"
        ))

        # convert the messages to a single string
        prompt = ""
        for message in messages:
            prompt += f"\n{message['role']}: {action}"
        return action, prompt