import { useState, useEffect, useRef } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Load saved theme (using state instead of localStorage)
  useEffect(() => {
    // Default to dark mode for better appearance
    setDarkMode(true);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message 
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    try {
      // Call FastAPI backend
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: currentInput }),
      });
      const data = await res.json();

      // Add assistant message
      setMessages([...newMessages, { role: "assistant", content: data.answer }]);
    } catch (error) {
      // Handle error gracefully
      setMessages([...newMessages, { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex flex-col h-screen transition-all duration-500 ease-in-out ${
      darkMode 
        ? "bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" 
        : "bg-gradient-to-br from-blue-50 via-white to-indigo-50"
    }`}>
      {/* Header with theme toggle */}
      <div className={`flex justify-between items-center p-6 border-b backdrop-blur-sm transition-all duration-500 ease-in-out ${
        darkMode 
          ? "bg-slate-800/50 border-slate-700 text-white" 
          : "bg-white/70 border-gray-200 text-gray-800"
      }`}>
        <div>
          <h1 className="text-2xl font-bold">HealthLLM Chat</h1>
          <p className={`text-sm ${darkMode ? "text-slate-400" : "text-gray-600"}`}>
            Ask me anything about health
          </p>
        </div>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`px-4 py-2 rounded-xl font-medium transition-all duration-500 ease-in-out hover:scale-105 ${
            darkMode
              ? "bg-gradient-to-r from-yellow-400 to-orange-400 text-gray-900 hover:from-yellow-300 hover:to-orange-300"
              : "bg-gradient-to-r from-slate-700 to-slate-900 text-white hover:from-slate-600 hover:to-slate-800"
          }`}
        >
          {darkMode ? "☀️ Light" : "🌙 Dark"}
        </button>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 transition-all duration-500 ease-in-out">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className={`text-center p-8 rounded-2xl transition-all duration-500 ease-in-out ${
              darkMode ? "bg-slate-800/50 text-slate-400" : "bg-white/50 text-gray-500"
            }`}>
              <div className="text-6xl mb-4">💬</div>
              <h3 className="text-xl font-semibold mb-2">Start a conversation</h3>
              <p>Ask me anything about health and wellness</p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div className={`flex items-start space-x-3 max-w-[85%] md:max-w-[70%]`}>
              {msg.role === "assistant" && (
                <div className="flex flex-col items-center space-y-1">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-500 ease-in-out ${
                    darkMode ? "bg-emerald-600 text-white" : "bg-emerald-500 text-white"
                  }`}>
                    🏥
                  </div>
                  <span className={`text-xs font-medium transition-all duration-500 ease-in-out ${
                    darkMode ? "text-slate-400" : "text-gray-500"
                  }`}>
                    HealthLLM
                  </span>
                </div>
              )}
              
              <div
                className={`px-4 py-3 rounded-2xl shadow-sm break-words transition-all duration-500 ease-in-out ${
                  msg.role === "user"
                    ? darkMode
                      ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-br-md"
                      : "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md"
                    : darkMode
                      ? "bg-slate-700/80 text-slate-100 rounded-bl-md border border-slate-600"
                      : "bg-white text-gray-800 rounded-bl-md border border-gray-200"
                }`}
              >
                <div className="whitespace-pre-wrap leading-relaxed transition-all duration-500 ease-in-out">
                  {msg.content}
                </div>
              </div>

              {msg.role === "user" && (
                <div className="flex flex-col items-center space-y-1">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-500 ease-in-out ${
                    darkMode ? "bg-blue-600 text-white" : "bg-blue-500 text-white"
                  }`}>
                    👤
                  </div>
                  <span className={`text-xs font-medium transition-all duration-500 ease-in-out ${
                    darkMode ? "text-slate-400" : "text-gray-500"
                  }`}>
                    You
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-start space-x-3 max-w-[85%] md:max-w-[70%]">
              <div className="flex flex-col items-center space-y-1">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-500 ease-in-out ${
                  darkMode ? "bg-emerald-600 text-white" : "bg-emerald-500 text-white"
                }`}>
                  🏥
                </div>
                <span className={`text-xs font-medium transition-all duration-500 ease-in-out ${
                  darkMode ? "text-slate-400" : "text-gray-500"
                }`}>
                  HealthLLM
                </span>
              </div>
              <div className={`px-4 py-3 rounded-2xl rounded-bl-md transition-all duration-500 ease-in-out ${
                darkMode ? "bg-slate-700/80 border border-slate-600" : "bg-white border border-gray-200"
              }`}>
                <div className="flex space-x-1 transition-all duration-500 ease-in-out">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${
                    darkMode ? "bg-slate-400" : "bg-gray-400"
                  }`}></div>
                  <div className={`w-2 h-2 rounded-full animate-pulse delay-75 ${
                    darkMode ? "bg-slate-400" : "bg-gray-400"
                  }`}></div>
                  <div className={`w-2 h-2 rounded-full animate-pulse delay-150 ${
                    darkMode ? "bg-slate-400" : "bg-gray-400"
                  }`}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input form */}
      <div className={`p-4 border-t backdrop-blur-sm transition-all duration-500 ease-in-out ${
        darkMode 
          ? "bg-slate-800/50 border-slate-700" 
          : "bg-white/70 border-gray-200"
      }`}>
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className={`w-full px-4 py-3 rounded-xl border focus:outline-none focus:ring-2 transition-all duration-500 ease-in-out ${
                darkMode
                  ? "bg-slate-700 border-slate-600 text-white placeholder-slate-400 focus:ring-blue-500 focus:border-transparent"
                  : "bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:ring-blue-500 focus:border-transparent"
              }`}
              placeholder="Ask about symptoms, conditions, treatments..."
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage(e)}
              disabled={isLoading}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className={`px-6 py-3 rounded-xl font-medium transition-all duration-500 ease-in-out hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 ${
              darkMode
                ? "bg-gradient-to-r from-emerald-600 to-emerald-700 text-white hover:from-emerald-500 hover:to-emerald-600"
                : "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-400 hover:to-emerald-500"
            }`}
          >
            {isLoading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;