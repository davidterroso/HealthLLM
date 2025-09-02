import { useState, useEffect, useRef } from "react";
import Typewriter from "./Typewritter";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    setDarkMode(true);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Trigger transition animation if first message
    if (messages.length === 0) {
      setIsTransitioning(true);
      // Start hiding welcome screen
      setTimeout(() => setShowWelcome(false), 300);
    }

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
      
      // Complete transition
      if (isTransitioning) {
        setIsTransitioning(false);
      }
    } catch (error) {
      // Handle error gracefully
      setMessages([...newMessages, { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
      }]);
      
      if (isTransitioning) {
        setIsTransitioning(false);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen relative overflow-hidden">
      {/* Background layers for smooth transitions */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-indigo-50 transition-opacity duration-500 ease-in-out"
           style={{ opacity: darkMode ? 0 : 1 }}></div>
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 transition-opacity duration-500 ease-in-out"
           style={{ opacity: darkMode ? 1 : 0 }}></div>
      
      {/* Content wrapper */}
      <div className="relative z-10 flex flex-col h-full">
        {/* Header with theme toggle */}
        <div className={`flex justify-between items-center p-6 border-b backdrop-blur-sm transition-all duration-500 ease-in-out ${
          darkMode 
            ? "bg-slate-800/50 border-slate-700 text-white" 
            : "bg-white/70 border-gray-200 text-gray-800"
        }`}>
          <div>
            <h1 className="text-2xl font-bold">HealthLLM</h1>
            <p className={`text-sm ${darkMode ? "text-slate-400" : "text-gray-600"}`}>
              Your medical advisor
            </p>
          </div>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`px-4 py-2 rounded-xl font-medium transition-all duration-500 ease-in-out hover:scale-105 active:scale-95 ${
              darkMode
                ? "bg-gradient-to-r from-yellow-400 to-orange-400 text-gray-900 hover:from-yellow-300 hover:to-orange-300"
                : "bg-gradient-to-r from-slate-700 to-slate-900 text-white hover:from-slate-600 hover:to-slate-800"
            }`}
          >
            {darkMode ? "☀️ Light" : "🌙 Dark"}
          </button>
        </div>

        {/* Messages Container */}
          <div
            className={`flex-1 p-4 space-y-4 relative transition-all duration-500 ease-in-out
              scrollbar-thin scrollbar-thumb-rounded-xl 
              ${darkMode 
                ? "bg-gradient-to-br from-slate-900/20 via-slate-800/20 to-slate-900/20 scrollbar-thumb-slate-700 scrollbar-track-slate-800/40" 
                : "bg-gradient-to-br from-blue-50/30 via-white/30 to-indigo-50/30 scrollbar-thumb-blue-400 scrollbar-track-blue-100"
              }`}
            style={{ overflowY: showWelcome ? "hidden" : "auto" }}
          >
          {/* Welcome Screen with smooth exit animation */}
          {showWelcome && (
            <div className={`absolute inset-0 flex items-center justify-center welcome-screen transition-all duration-700 ease-out ${
              isTransitioning 
                ? 'opacity-0 transform scale-95 translate-y-8' 
                : 'opacity-100 transform scale-100 translate-y-0'
            }`}>
              <div className={`text-center p-8 rounded-2xl transition-all duration-500 ease-in-out ${
                darkMode ? "bg-slate-800/50 text-slate-400" : "bg-white/50 text-gray-500"
              }`}>
                <Typewriter text="HealthLLM" />
                <h3 className="text-xl font-semibold mt-4 mb-2">Start a conversation</h3>
                <p>Ask me anything about health and wellness</p>
              </div>
            </div>
          )}

          {/* Chat Messages with smooth entrance animation */}
          <div className={`transition-all duration-700 ease-out ${
            messages.length > 0 
              ? 'opacity-100 transform translate-y-0' 
              : 'opacity-0 transform translate-y-4'
          }`}>
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} ${
                  i === 0 ? 'animate-fade-in-up' : ''
                }`}
                style={{
                  animationDelay: i === 0 ? '300ms' : '0ms'
                }}
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
          </div>

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex justify-start animate-fade-in">
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
            className={`inline-flex items-center justify-center
                        w-28 shrink-0 flex-none text-center
                        px-6 py-3 rounded-xl font-medium
                        transition-transform duration-200 ease-out
                        hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
                        ${
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
    </div>
  );
}

export default App;