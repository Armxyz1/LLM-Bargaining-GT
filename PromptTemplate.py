from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, HuggingFaceHub  # Add other model integrations as needed

# Define a prompt template
prompt_template_1 = PromptTemplate(
    input_variables=["description_1", "goal_1", "history_1"],
    template=(
        "You are a rational and intelligent agent participating in the following bargaining scenario with another rational and intelligent agent: {description_1}. "
        "As Player 1, your objective is: {goal_1}. "
        "Here is the conversation history so far: {history_1}. "
        "Your response as Player 1 is:"
    )
)

prompt_template_2 = PromptTemplate(
    input_variables=["description_2", "goal_2", "history_2"],
    template=(
        "You are a rational and intelligent agent participating in the following bargaining scenario with another rational and intelligent agent: {description_2}. "
        "As Player 2, your objective is: {goal_2}. "
        "Here is the conversation history so far: {history_2}. "
        "Your response as Player 2 is:"
    )
)

# Create memory for storing conversation history
memory_agent1 = ConversationBufferMemory()
memory_agent2 = ConversationBufferMemory()

# Function to initialize an agent with a specified model
def initialize_agent(model_name, prompt_template, memory):
    if model_name == "OpenAI-GPT":
        llm = ChatOpenAI(temperature=0.7)
    elif model_name == "LLama":
        llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature": 0.7})
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return ConversationChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
    )

# Initialize agents with the desired models
agent1 = initialize_agent("OpenAI-GPT", prompt_template_1, memory_agent1)
agent2 = initialize_agent("LLama", prompt_template_2, memory_agent2)

# Function to facilitate interaction between two agents
def agent_interaction(agent1, agent2, turns, description, goal_1, goal_2):
    history_1 = ""
    history_2 = ""
    
    for turn in range(turns):
        # Agent 1's turn
        response_1 = agent1.run(
            description_1=description,
            goal_1=goal_1,
            history_1=history_1,
        )
        history_1 += f"Agent 1: {response_1}\n"
        
        # Agent 2's turn
        response_2 = agent2.run(
            description_2=description,
            goal_2=goal_2,
            history_2=history_2,
        )
        history_2 += f"Agent 2: {response_2}\n"
        
        print(f"Turn {turn + 1}:\nAgent 1: {response_1}\nAgent 2: {response_2}\n")

# Start interaction
agent_interaction(agent1, agent2)

if __name__ == "__main__":
    description = ""
    goal_1 = ""
    goal_2 = ""
    turns = 5  # Number of turns for the interaction
    agent_interaction(agent1, agent2, turns, description, goal_1, goal_2)