import React, { useState } from "react";
import { getBezierPath, EdgeText, getMarkerEnd } from "reactflow";
import "./CustomEdge.css";

const CustomEdge = ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    data,
    markerEnd: defaultMarkerEnd,
    label,
    animated: defaultAnimated,
}) => {
    const [isHovered, setIsHovered] = useState(false);

    // Extract weight from the label or data
    const extractWeight = () => {
        if (data && data.weight !== undefined) {
            return data.weight;
        }

        // If label contains weight in format "type (weight)"
        if (label && typeof label === "string") {
            const match = label.match(/\(([0-9.]+)\)$/);
            if (match && match[1]) {
                return parseFloat(match[1]);
            }
        }

        return 1; // Default weight if not found
    };

    const weight = extractWeight();

    // Calculate opacity based on weight (assuming weight is between 0 and 1)
    // If weight is outside this range, we need to normalize it
    const normalizeWeight = (w) => {
        // Assuming weights are between 0 and 1
        // If they're in a different range, adjust this function
        return Math.min(Math.max(w, 0.1), 1);
    };

    const opacity = normalizeWeight(weight);

    // Always animate edges to show direction
    const animated = true;

    // Get the path for the edge
    const [edgePath, labelX, labelY] = getBezierPath({
        sourceX,
        sourceY,
        sourcePosition,
        targetX,
        targetY,
        targetPosition,
    });

    // Ensure we have a marker end for direction
    const markerEnd = getMarkerEnd("arrow", defaultMarkerEnd);

    return (
        <>
            <path
                id={id}
                style={{
                    ...style,
                    strokeOpacity: opacity,
                    stroke: style.stroke || "#b1b1b7",
                    strokeWidth: 1.5,
                }}
                className={`react-flow__edge-path ${
                    animated ? "animated-edge-path" : ""
                }`}
                d={edgePath}
                markerEnd={markerEnd}
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
            />

            {isHovered && label && (
                <EdgeText
                    x={labelX}
                    y={labelY}
                    label={label}
                    labelStyle={{ fill: "#888", fontWeight: 500 }}
                    labelShowBg
                    labelBgStyle={{ fill: "white", fillOpacity: 0.75, rx: 4 }}
                />
            )}
        </>
    );
};

export default CustomEdge;
