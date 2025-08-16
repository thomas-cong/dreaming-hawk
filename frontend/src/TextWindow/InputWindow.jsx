import React from "react";
import useEditor from "./useEditor";
import { Editor, RichUtils } from "draft-js";
import "./InputWindow.css";

const InputWindow = () => {
    const { editorState, setEditorState } = useEditor();

    const handleKeyCommand = (command, editorState) => {
        const newState = RichUtils.handleKeyCommand(editorState, command);
        if (newState) {
            setEditorState(newState);
            return "handled";
        }
        return "not-handled";
    };

    const onBoldClick = () => {
        setEditorState(RichUtils.toggleInlineStyle(editorState, "BOLD"));
    };

    const onItalicClick = () => {
        setEditorState(RichUtils.toggleInlineStyle(editorState, "ITALIC"));
    };

    const onUnderlineClick = () => {
        setEditorState(RichUtils.toggleInlineStyle(editorState, "UNDERLINE"));
    };

    return (
        <div className="editor-container">
            <div className="toolbar">
                <button onClick={onBoldClick}>Bold</button>
                <button onClick={onItalicClick}>Italic</button>
                <button onClick={onUnderlineClick}>Underline</button>
            </div>
            <div className="editor">
                <Editor
                    editorState={editorState}
                    onChange={setEditorState}
                    handleKeyCommand={handleKeyCommand}
                    placeholder="Type something..."
                />
            </div>
        </div>
    );
};

export default InputWindow;
