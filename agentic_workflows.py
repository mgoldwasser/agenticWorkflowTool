from agents import Agent, Runner  # Removed AgentRunConfig import
from agents.tool import function_tool
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
import openai
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_openai_response(response):
    logging.info(f"OpenAI Response: {response}")

# Model for defining a generic agent
class AgentDefinition(BaseModel):
    name: str
    instructions: str
    tools: List[Any] = []
    handoffs: Optional[List[Any]] = None  # FIX: Ensure handoffs is optional and handled correctly

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow arbitrary types  # FIX: Allow arbitrary types

# Function to dynamically create agents from definitions
def create_agents(definitions: Dict[str, AgentDefinition]) -> Dict[str, Agent]:
    created_agents = {}
    for key, definition in definitions.items():
        created_agents[key] = Agent(
            name=definition.name,
            instructions=definition.instructions,
            tools=definition.tools,
            handoffs=definition.handoffs or []  # FIX: Ensure handoffs is a list, even if None
        )
    return created_agents

# Generic agent definitions
agent_definitions = {
    "string_output_agent": AgentDefinition(
        name="string_output_agent",
        instructions="Provide outputs in simple string format based on given prompt."
    ),
    "list_output_agent": AgentDefinition(
        name="list_output_agent",
        instructions="Parse any given text input into a structured json array of strings."
    ),
    "reasoning_agent": AgentDefinition(
        name="reasoning_agent",
        instructions="Provide detailed reasoning and logical explanations based on the provided prompt."
    ),
    "user_prompt_agent": AgentDefinition(
        name="user_prompt_agent",
        instructions="Ask questions to the user to gather more information clearly and concisely based on given prompts."
    ),
    "end_workflow_agent": AgentDefinition(
        name="end_workflow_agent",
        instructions="Determine if the workflow is complete and end it if the goal has been achieved. Simply respond 'workflow complete.''"
    )
}

# Create generic agents
generated_agents = create_agents(agent_definitions)

# Example tool definition
@function_tool
def parse_to_list(input_text: str) -> List[str]:
    return [item.strip() for item in input_text.split('\n') if item.strip()]

# Function to prompt the user interactively
async def user_prompt_interaction(prompt_agent: Agent, prompt: str, run_config: Any) -> str:
    prompt_question_output = await Runner.run(
        prompt_agent,
        [{'role': 'user', 'type': 'message', 'content': prompt}]
    )
    user_response = input(f"{prompt_question_output.final_output}\n> ")
    return user_response

# Triage orchestrator function to dynamically select agent
async def orchestrator(prompt: str, run_config: Any):
    retry_attempts = 3  # Max retries for rate limits
    for attempt in range(retry_attempts):
        try:
            triage_agent = Agent(
                name="triage_agent",
                instructions="Determine the next step in the workflow based on the goal and task status.",
                handoffs=[
                    generated_agents["string_output_agent"],
                    generated_agents["list_output_agent"],
                    generated_agents["reasoning_agent"],
                    generated_agents["user_prompt_agent"],
                    generated_agents["end_workflow_agent"]
                ]
            )
            logging.info(f"Sending prompt to OpenAI: {str(prompt)[:80]}...")
            output = await Runner.run(triage_agent, [{'role': 'user', 'type': 'message', 'content': prompt}])
            log_openai_response(output.final_output)
            return output.final_output
        except openai.RateLimitError as e:
            wait_time = 60  # Default wait time if not provided by API
            if 'error' in e.args[0] and 'message' in e.args[0]['error']:
                message = e.args[0]['error']['message']
                if 'try again after' in message:
                    wait_time = float(message.split('try again after ')[-1].split()[0])
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except openai.APIConnectionError:
            print("Connection error. Retrying in 5 seconds...")
            time.sleep(5)
    raise RuntimeError("Failed after multiple retries due to API limits or connection errors")

def default_task_formatter(task: str) -> str:
    return f"Process this task: '{task}'"

# Workflow execution logic (example of dynamic usage)
async def run_dynamic_workflow():
    config = {}

    # Get user goal at the beginning
    user_goal = await user_prompt_interaction(
        generated_agents["user_prompt_agent"],
        "What would you like to accomplish?",
        config
    )

    # Context tracking per task
    task_contexts = {user_goal: [user_goal]}
    tasks = [user_goal]
    completed_tasks = set()  # Track completed tasks to prevent loops
    
    while tasks:
        current_task = tasks.pop(0)

        # Skip tasks that have already been processed
        if current_task in completed_tasks:
            continue

        context = "\n".join(task_contexts.get(current_task, []))  # Get only relevant context
        task_output = await orchestrator(context, config)

        # Log and check workflow completion
        if "workflow complete" in task_output.lower():
            logging.info("Workflow completed successfully.")
            break

        try:
            cleaned_output = re.sub(r'```[a-zA-Z]*\n|\n```', '', task_output).strip()
            new_tasks = json.loads(cleaned_output)

            # If the output is not a list, convert it into one for consistency
            if isinstance(new_tasks, str):
                new_tasks = [new_tasks]

            if not new_tasks:
                logging.info("No further tasks generated, ending workflow.")
                break  # No new tasks means the workflow is done

            # Handle merging of parallel tasks correctly
            merged_output = []
            for new_task in new_tasks:
                if new_task in completed_tasks or new_task in tasks:
                    continue  # Skip if already processed
                
                # Add only the most relevant context
                task_contexts[new_task] = list(set(task_contexts[current_task] + [f"Task: {new_task}"]))
                if new_task not in completed_tasks and new_task not in tasks:
                    tasks.append(new_task)
                    merged_output.append(new_task)

            # Log merging of parallel tasks correctly
            if len(merged_output) > 1:
                logging.info(f"Merging parallel tasks: {merged_output}")

            # Mark task as completed
            completed_tasks.add(current_task)

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}. Continuing with remaining tasks.")
    
    logging.info("All tasks processed. Calling end_workflow_agent...")
    await orchestrator("End the workflow.", config)


# Example execution
if __name__ == "__main__":
    async def main():
        await run_dynamic_workflow()

    asyncio.run(main())  # Ensures proper execution in an async loop

