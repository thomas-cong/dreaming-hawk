import React, { useState, useEffect } from "react";
import axios from "axios";

// --- Configuration ---
// The base URL for your Python backend API.
// Make sure your FastAPI backend is running and accessible at this address.
const API_BASE_URL = "http://localhost:8000";

/**
 * A minimal and clean main application component.
 * It demonstrates how to fetch data from the backend on component mount.
 */
function App() {
    // --- State ---
    // `backendStatus` stores the health status message from the backend.
    // `error` stores any error message if the API call fails.
    const [backendStatus, setBackendStatus] = useState(
        "Checking backend status..."
    );
    const [error, setError] = useState(null);

    // --- Effects ---
    // `useEffect` with an empty dependency array `[]` runs once when the component mounts.
    // This is the perfect place to fetch initial data.
    useEffect(() => {
        // Define an async function to fetch the health status.
        const checkBackendHealth = async () => {
            try {
                // Make a GET request to the /health endpoint.
                const response = await axios.get(`${API_BASE_URL}/health`);

                // If the request is successful, update the state with the status.
                // The backend returns { "status": "ok" }, so we access `response.data.status`.
                setBackendStatus(`Backend is running: ${response.data.status}`);
                setError(null); // Clear any previous errors
            } catch (err) {
                // If the request fails, update the state with an error message.
                console.error("Failed to connect to backend:", err);
                setBackendStatus("Could not connect to backend.");
                setError(
                    "Please ensure the Python backend is running on " +
                        API_BASE_URL
                );
            }
        };

        // Call the function to execute the health check.
        checkBackendHealth();
    }, []); // Empty dependency array ensures this runs only once.

    // --- Render ---
    // This is the JSX that defines the component's UI.
    return (
        <div
            style={{
                fontFamily: "sans-serif",
                textAlign: "center",
                marginTop: "50px",
            }}
        >
            <header>
                <h1>React Frontend Skeleton</h1>
                <p>A minimal starting point for your application.</p>
            </header>
            <main style={{ marginTop: "30px" }}>
                <h2>Backend Connection Test</h2>
                <p>{backendStatus}</p>
                {error && <p style={{ color: "red" }}>{error}</p>}
            </main>
            <footer style={{ marginTop: "50px", color: "#888" }}>
                <p>You can now start building your features!</p>
                <p>
                    Edit <code>src/App.jsx</code> to get started.
                </p>
            </footer>
        </div>
    );
}

export default App;
