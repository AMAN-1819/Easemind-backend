from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from agno.agent import Message
from app_folder.agents_folder.gita_agent import GitaAIAgent
from app_folder.agents_folder.article_agent import ArticleSuggestionAgent
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional



app = FastAPI(title="Gita Wisdom API", description="A FastAPI backend to get Gita advice and resources for stressed students.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for tighter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionRequest(BaseModel):
    question: str

# ChatCoordinator
class ChatCoordinator:
    def __init__ (self, agents):
        self.agents = {agent.name: agent for agent in agents}

    def handle_question(self, question):
        print("User Question:", question)

        gita_agent = self.agents["GitaAIAgent"]
        gita_answer = gita_agent.run(Message(role="user", content=question))

        #print("Gita Answer:", gita_answer.content)

        article_agent = self.agents["ArticleSuggestionAgent"]
        resources_answer = article_agent.run(gita_answer)

        #print("Resources Answer:", resources_answer.content)

        output = "Gita Tips:\n"
        gita_tips = gita_answer.content.strip().split("\n")
        for tip in gita_tips:
            output += f"{tip}\n"

        output += "\nResources:\n"
        output += resources_answer.content

        return output



# Initialize agents and coordinator
gita_agent = GitaAIAgent()
article_agent = ArticleSuggestionAgent()
coordinator = ChatCoordinator([gita_agent, article_agent])

# FastAPI endpoint for POST
@app.post("/question", response_model=dict)
async def get_advice(request: QuestionRequest):
    try:
        question = request.question
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        answer = coordinator.handle_question(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
@app.get("/")
def read_root():
    return {"message": "Gita AI Assistant is running"}

# FastAPI endpoint for GET (added from old code)
@app.get("/api/ask")
async def ask_gita_get(
    question: Optional[str] = Query(None),
    q: Optional[str] = Query(None, alias="question")
):
    query = question or q
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid question")
    
    try:
        answer = coordinator.handle_question(query)
        return {
            "question": query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the server programmatically
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)