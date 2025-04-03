from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Define a prompt template
prompt_template_1 = PromptTemplate(
    input_variables=["description", "goal", "history", "input"],
    template=(
        "You are a rational and intelligent agent. You are part of the following bargain scenario: {description}. "
        "Your goal is to: {goal}. Conversation history: {history}. "
        "Player 1: {input}. "
    )
)

prompt_template_2 = PromptTemplate(
    input_variables=["description", "goal", "history", "input"],
    template=(
        "You are a rational and intelligent agent. You are part of the following bargain scenario: {description}. "
        "Your goal is to: {goal}. Conversation history: {history}. "
        "Player 2: {input}. "
    )
)

# Create memory for storing conversation history
memory_agent1 = ConversationBufferMemory()
memory_agent2 = ConversationBufferMemory()

# Initialize two agents with the same prompt template but separate memory
agent1 = ConversationChain(
    llm=ChatOpenAI(temperature=0.7),
    prompt=prompt_template_1,
    memory=memory_agent1,
)

agent2 = ConversationChain(
    llm=ChatOpenAI(temperature=0.7),
    prompt=prompt_template_2,
    memory=memory_agent2,
)

# Function to facilitate interaction between two agents
def agent_interaction(agent1, agent2, initial_message, turns=5):
    message = initial_message
    for _ in range(turns):
        print(f"Agent 1: {message}")
        response_agent2 = agent2.run(input=message)
        print(f"Agent 2: {response_agent2}")
        message = response_agent2
        response_agent1 = agent1.run(input=message)
        print(f"Agent 1: {response_agent1}")
        message = response_agent1

# Start interaction
agent_interaction(agent1, agent2, initial_message="Hello, how are you?")