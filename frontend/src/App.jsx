import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! I am your local OSTEP (Operating Systems: Three Easy Pieces) Assistant. What operating systems concept can I help you with today?"
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Automatically scroll to the newest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: userQuery }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: Could not connect to the backend. Is FastAPI running?" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-title">
          <span className="logo">⚡</span> OSTEP 
          <span style={{ fontSize: "0.85rem", color: "#8b949e", marginLeft: "8px", fontWeight: "400" }}>
            (Operating Systems: Three Easy Pieces)
          </span> 
          RAG System
        </div>
        <div className="header-status">
          <span className="status-dot"></span> Online
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-row ${msg.role}`}>
            <div className={`message-bubble ${msg.role}`}>
              <div className="message-role">{msg.role === "assistant" ? "OSTEP AI" : "You"}</div>
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message-row assistant">
            <div className="message-bubble assistant loading-bubble">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the textbook..."
            disabled={isLoading}
            className="chat-input"
          />
          <button type="submit" disabled={!input.trim() || isLoading} className="send-button">
            Send
          </button>
        </form>
        <div className="footer-note">Running locally on Qwen2.5-7B • RTX 5060 Optimized</div>
      </footer>
    </div>
  );
}

export default App;