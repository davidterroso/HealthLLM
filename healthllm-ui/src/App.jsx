import { useState, useEffect } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);

  // Load saved theme
  useEffect(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "dark") setDarkMode(true);
  }, []);

  // Apply theme on change
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

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
    <div className="flex flex-col h-screen p-4 bg-gray-100 dark:bg-gray-900 text-black dark:text-white transition-colors duration-300">
      {/* Header with theme toggle */}
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-xl font-bold">HealthLLM Chat</h1>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="px-3 py-1 rounded-lg bg-gray-300 dark:bg-gray-700"
        >
          {darkMode ? "🌞 Light" : "🌙 Dark"}
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-2 rounded-lg max-w-md ${
              msg.role === "user"
                ? "bg-blue-500 text-white self-end ml-auto"
                : "bg-gray-300 dark:bg-gray-700"
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>

      {/* Input form */}
      <form onSubmit={sendMessage} className="flex mt-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 border rounded-lg p-2 dark:bg-gray-800 dark:border-gray-600"
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
