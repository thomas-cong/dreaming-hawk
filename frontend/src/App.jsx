import React, { useState, useEffect, useRef, useMemo } from "react";
import ReactFlow, {
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
} from "reactflow";
import "reactflow/dist/style.css";

const simpleHash = (str) => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};

const getPositionFromLemma = (lemma) => {
    const hash = simpleHash(lemma);
    const x = (hash & 0xffff) % 800;
    const y = (hash >> 16) % 600;
    return { x, y };
};

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
                ws.current.send(JSON.stringify({ text: newText, mode: "add" }));
            }
            setCurrentWord("");
        } else {
            const words = newText.trim().split(/\s+/);
            setCurrentWord(words[words.length - 1] || "");
        }
    };

    const updateGraph = (message) => {
        if (message.type === 'full') {
            const { nodes: incomingNodes, edges: incomingEdges } = message.payload;

            const newNodes = incomingNodes.map((node) => {
                const lemma = node.data.lemmatized?.[0] || node.data.word;
                const position = getPositionFromLemma(lemma);
                return {
                    id: node.id.toString(),
                    data: { label: `${node.data.word} (${node.data.value})` },
                    position,
                };
            });

            const newEdges = incomingEdges.map((link) => ({
                id: `${link.source}-${link.target}-${link.key}`,
                source: link.source.toString(),
                target: link.target.toString(),
                label: `${link.type} (${link.weight.toFixed(2)})`,
                style: {
                    stroke: link.type === "semantic" ? "#4ade80" : "#60a5fa",
                },
            }));

            setNodes(newNodes);
            setEdges(newEdges);
        } else if (message.type === 'diff') {
            const { added_nodes, updated_nodes, added_edges, updated_edges } = message.payload;

            const nodeUpdates = [...added_nodes, ...updated_nodes].map((node) => {
                const lemma = node.data.lemmatized?.[0] || node.data.word;
                const position = getPositionFromLemma(lemma);
                return {
                    id: node.id.toString(),
                    data: { label: `${node.data.word} (${node.data.value})` },
                    position,
                };
            });

            const edgeUpdates = [...added_edges, ...updated_edges].map((link) => ({
                id: `${link.source}-${link.target}-${link.key}`,
                source: link.source.toString(),
                target: link.target.toString(),
                label: `${link.type} (${link.weight.toFixed(2)})`,
                style: {
                    stroke: link.type === "semantic" ? "#4ade80" : "#60a5fa",
                },
            }));

            setNodes(currentNodes => {
                const nodeMap = new Map(currentNodes.map(n => [n.id, n]));
                nodeUpdates.forEach(n => nodeMap.set(n.id, n));
                return Array.from(nodeMap.values());
            });

            setEdges(currentEdges => {
                const edgeMap = new Map(currentEdges.map(e => [e.id, e]));
                edgeUpdates.forEach(e => edgeMap.set(e.id, e));
                return Array.from(edgeMap.values());
            });
        }
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
