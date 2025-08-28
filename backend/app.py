from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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
    wg = WordGraph(text_window_size=30, semantic_threshold=0.6)
    try:
        while True:
            data = await websocket.receive_text()
            
            # The generator yields graph states
            graph_generator = wg.add_text(data, yield_frames=True, frame_step=1, reset_window=True)
            
            for graph_state in graph_generator:
                json_data = graph_state.jsonify()
                await websocket.send_text(json_data)
                # Add a small delay to prevent overwhelming the client and allow for smooth visualization.
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011)