from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from Graphs import WordGraph

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

wg = WordGraph(text_window_size=5)

@app.get("/test")
def test():
    return {"status": "ok"}
@app.post("/reset")
def reset():
    global wg
    wg = WordGraph(text_window_size=5)
    return {"status": "ok"}
@app.post("/add_text")
def add_text(text: str):
    wg.add_text(text)
    return {"status": "ok"}
