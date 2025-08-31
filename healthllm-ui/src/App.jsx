// src/App.jsx
import { useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async (e) => {
    e.preventDefault();

    // Add user message
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");

    // Call FastAPI backend
    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: input }),
    });
    const data = await res.json();

    // Add assistant message
    setMessages([...newMessages, { role: "assistant", content: data.answer }]);
  };

  return (
    <div className="flex flex-col h-screen p-4 bg-gray-100">
      <div className="flex-1 overflow-y-auto space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-2 rounded-lg max-w-md ${
              msg.role === "user" ? "bg-blue-500 text-white self-end" : "bg-gray-300"
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>

      <form onSubmit={sendMessage} className="flex mt-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 border rounded-lg p-2"
          placeholder="Type your message..."
        />
        <button
          type="submit"
          className="ml-2 px-4 py-2 bg-green-500 text-white rounded-lg"
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
