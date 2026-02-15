import asyncio
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition


from config.config import GROQ_API_KEY
from agent.tools import (
    get_n_random_words
)
from utils.logger import get_logger

logger = get_logger(__name__)

# The agent state : (Act's like a Short term memory for agent)
class AgentState(TypedDict):
    messages : Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]


# Tools
local_tools = [
    get_n_random_words,
]


async def setup_tools():
    return [*local_tools]


# Assistant
def assistant(state : AgentState):

    textual_description_of_tools = """
        def get_n_random_words(language: str,
                       n: int, ) -> list:
        Selects a specified number of random words from a language-specific word list.

        The function reads a JSON file containing words for the specified language from
        a predefined directory. It then selects `n` random words from the file and
        returns them in a list.

        :param language: A string representing the language for which to fetch the word list.
        :param n: An integer specifying the number of random words to retrieve.
        :return: A list containing `n` randomly selected words.
    """
    
    sys_msg = SystemMessage(content = f"""
        You are a helpful language learning assistant. You can carry out actions using the following tools: {textual_description_of_tools}. 

        The user is going to give you a command.

        Your job is to check:
        1. Which source language the user wants words from.
        2. How many words they want.

        Here are some example workflows:
        input: Get 20 random words in Spanish.
        source language: Spanish
        number of words: 20
        
        input: Get 10 hard words in German
        source language: German
        number of words: 10                    
    """)

    # LLM
    tools = assistant.tools if hasattr(assistant, "tools") else []
    llm = ChatGroq(
        groq_api_key = GROQ_API_KEY,
        model_name = "llama-3.3-70b-versatile"
        )
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls = False)

    return {
        "messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state["source_language"],
        "number_of_words": state["number_of_words"],
    }


async def build_graph():
    """Build the state graph with properly initialized tools."""

    tools = await setup_tools()
    assistant.tools = tools

    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition
        )
    builder.add_edge("tools", "assistant")

    return builder.compile()


async def main():
    """Main async function to run the application."""

    react_garph = await build_graph()

    user_prompt = "Please get 10 random words in German."

    messages = [HumanMessage(content = user_prompt)]

    result = await react_garph.ainvoke({
        "messages" : messages,
        "source_language" : None,
        "number_of_words" : None
    })

    logger.info(f"Final messages : {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())