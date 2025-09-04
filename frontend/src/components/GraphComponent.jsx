import React, { useCallback, useEffect, useState } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Panel
} from "reactflow";
import ELK from 'elkjs/lib/elk.bundled.js';
import "reactflow/dist/style.css";
import "./GraphComponent.css";

const elk = new ELK();

// Layout options for ELK
const elkOptions = {
  'elk.algorithm': 'layered',
  'elk.layered.spacing.nodeNodeBetweenLayers': '100',
  'elk.spacing.nodeNode': '80',
  'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
  'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF'
};

const GraphComponent = ({ nodes, edges, onNodesChange, onEdgesChange }) => {
  const [layoutedNodes, setLayoutedNodes] = useState(nodes);
  const [layoutedEdges, setLayoutedEdges] = useState(edges);
  const [isLayouting, setIsLayouting] = useState(false);

  const getLayoutedElements = useCallback(async (nodes, edges) => {
    if (!nodes.length) return { nodes, edges };
    
    setIsLayouting(true);

    // Convert the nodes and edges to the format required by ELK
    const elkGraph = {
      id: 'root',
      layoutOptions: elkOptions,
      children: nodes.map(node => ({
        id: node.id,
        width: 150, // Default node width
        height: 50,  // Default node height
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        sources: [edge.source],
        targets: [edge.target],
      })),
    };

    try {
      const layoutedGraph = await elk.layout(elkGraph);
      
      // Apply the layout to the nodes
      const layoutedNodes = nodes.map(node => {
        const elkNode = layoutedGraph.children.find(n => n.id === node.id);
        if (elkNode) {
          return {
            ...node,
            position: {
              x: elkNode.x,
              y: elkNode.y,
            },
          };
        }
        return node;
      });

      setIsLayouting(false);
      return { nodes: layoutedNodes, edges };
    } catch (error) {
      console.error('ELK layout error:', error);
      setIsLayouting(false);
      return { nodes, edges };
    }
  }, []);

  useEffect(() => {
    // Apply layout when nodes or edges change
    getLayoutedElements(nodes, edges).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      setLayoutedNodes(layoutedNodes);
      setLayoutedEdges(layoutedEdges);
    });
  }, [nodes, edges, getLayoutedElements]);

  const handleLayout = useCallback(() => {
    getLayoutedElements(nodes, edges).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      setLayoutedNodes(layoutedNodes);
      setLayoutedEdges(layoutedEdges);
    });
  }, [nodes, edges, getLayoutedElements]);

  return (
    <div className="graph-container">
      <ReactFlow
        nodes={layoutedNodes}
        edges={layoutedEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        nodesDraggable={true}
        elementsSelectable={true}
      >
        <Panel position="top-right">
          <button 
            className="layout-button"
            onClick={handleLayout} 
            disabled={isLayouting}
          >
            {isLayouting ? 'Optimizing Layout...' : 'Optimize Layout'}
          </button>
        </Panel>
        <MiniMap 
          nodeStrokeWidth={3}
          zoomable
          pannable
        />
        <Controls />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
    </div>
  );
};

export default GraphComponent;
