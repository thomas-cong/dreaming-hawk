import { useState } from "react";
import axios from "axios";
import "./App.css";

// Get the backend URL from the preload script
const BACKEND_URL = window.electronAPI.getBackendUrl();

function App() {
    const [text, setText] = useState("");
    const [response, setResponse] = useState("");

    const handleAddText = async () => {
        try {
            const res = await axios.post(`${BACKEND_URL}/add_text`, { text });
            setResponse(`Added text: ${JSON.stringify(res.data)}`);
        } catch (error) {
            setResponse(`Error: ${error.message}`);
        }
    };

    const handleReset = async () => {
        try {
            const res = await axios.post(`${BACKEND_URL}/reset`);
            setResponse(`Graph reset: ${JSON.stringify(res.data)}`);
        } catch (error) {
            setResponse(`Error: ${error.message}`);
        }
    };

    return (
        <div className="App">
            <h1>Dreaming Hawk</h1>
            <textarea
                rows="4"
                cols="50"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter text to add to the graph..."
            />
            <div>
                <button onClick={handleAddText}>Add Text</button>
                <button onClick={handleReset}>Reset Graph</button>
            </div>
            <pre>{response}</pre>
        </div>
    );
}

export default App;
