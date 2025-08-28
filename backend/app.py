from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
# Fix the import path to use relative import instead of absolute
from Graphs.wordGraph import WordGraph

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This global instance will be used by HTTP endpoints, but WebSocket will create its own.
global_wg = WordGraph(text_window_size=30)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    global global_wg
    global_wg = WordGraph(text_window_size=30)
    return {"status": "ok"}

@app.post("/add_text")
def add_text(text: str):
    global_wg.add_text(text)
    return {"status": "ok"}

@app.get("/get_graph")
def get_json_representation():
    return global_wg.jsonify()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Each client gets its own graph instance to manage state.
    wg = WordGraph(text_window_size=30, semantic_threshold=0.5)
    try:
        while True:
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            # Process the text completely, which updates the diffs internally
            wg.add_text(data["text"], yield_frames=False, reset_window=True)

            # Get the JSON representation of the diff
            json_diff = wg.jsonify_diff()
            print(json_diff)
            await websocket.send_text(json_diff)

            # Clear the diff for the next update
            wg.clear_diff()
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011)