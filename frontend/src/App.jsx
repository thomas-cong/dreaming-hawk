import React, { useState, useEffect, useRef, useMemo } from "react";
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
} from "reactflow";
import "reactflow/dist/style.css";

function App() {
    const [text, setText] = useState("");
    const [currentWord, setCurrentWord] = useState("");
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const ws = useRef(null);

    useEffect(() => {
        ws.current = new WebSocket("ws://localhost:8000/ws");
        ws.current.onopen = () => console.log("WebSocket connected");
        ws.current.onclose = () => console.log("WebSocket disconnected");

        ws.current.onmessage = (event) => {
            console.log("Received data:", event.data);
            const graphData = JSON.parse(event.data);
            updateGraph(graphData);
        };

        return () => {
            ws.current.close();
        };
    }, []);

    const handleTextChange = (e) => {
        const newText = e.target.value;
        setText(newText);

        if (newText.endsWith(" ") || newText.endsWith("\n")) {
            if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                ws.current.send(newText);
            }
            setCurrentWord("");
        } else {
            const words = newText.trim().split(/\s+/);
            setCurrentWord(words[words.length - 1] || "");
        }
    };

    const updateGraph = (graphData) => {
        const newNodes = graphData.nodes.map((node) => ({
            id: node.id.toString(),
            data: { label: `${node.data.word} (${node.data.value})` },
            position: { x: Math.random() * 800, y: Math.random() * 600 },
        }));

        const newEdges = graphData.edges.map((link) => ({
            id: `${link.source}-${link.target}-${link.key}`,
            source: link.source.toString(),
            target: link.target.toString(),
            label: `${link.type} (${link.weight.toFixed(2)})`,
            animated: link.type === "temporal",
            style: {
                stroke: link.type === "semantic" ? "#4ade80" : "#60a5fa",
            },
        }));

        setNodes(newNodes);
        setEdges(newEdges);
    };

    const displayedNodes = useMemo(() => {
        const baseNodes = nodes.filter((n) => n.id !== "current-word");
        if (currentWord) {
            return [
                ...baseNodes,
                {
                    id: "current-word",
                    data: { label: currentWord },
                    position: { x: 400, y: 300 }, // Center position
                    style: {
                        background: "rgba(255, 255, 255, 0.5)",
                        borderColor: "#999",
                    },
                },
            ];
        }
        return baseNodes;
    }, [nodes, currentWord]);

    return (
        <div className="app-container">
            <div className="editor-container">
                <textarea
                    value={text}
                    onChange={handleTextChange}
                    placeholder="Start typing here..."
                    className="text-editor"
                />
            </div>
            <div className="graph-container">
                <ReactFlow
                    nodes={displayedNodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    fitView
                >
                    <MiniMap />
                    <Controls />
                    <Background />
                </ReactFlow>
            </div>
        </div>
    );
}

export default App;
