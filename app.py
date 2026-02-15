import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from assistant_groq import build_graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Lumen Language Learning Agent API")

react_graph = None


class PromptRequest(BaseModel):
    prompt: str


@app.on_event("startup")
async def startup_event():
    global react_graph
    react_graph = await build_graph()


@app.post("/chat")
async def chat(req: PromptRequest):
    global react_graph

    messages = [HumanMessage(content=req.prompt)]

    result = await react_graph.ainvoke({
        "messages": messages,
        "source_language": None,
        "number_of_words": None,
        "word_difficulty": None,
        "target_language": None
    })

    return {"response": result["messages"][-1].content}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
