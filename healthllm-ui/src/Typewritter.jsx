import { useEffect, useState } from "react";

export default function Typewriter({ text }) {
  const [displayed, setDisplayed] = useState("");
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setDisplayed(text.slice(0, index + 1));
      index++;
      if (index === text.length) {
        clearInterval(interval);
        setShowCursor(true);
      }
    }, 150);

    return () => clearInterval(interval);
  }, [text]);

  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 500);

    return () => clearInterval(cursorInterval);
  }, []);

  return (
    <div className="flex justify-center">
      <div className="text-4xl font-mono font-bold tracking-wide mb-4 flex items-center">
        <span>{displayed}</span>
        <span 
          className={`ml-1 w-0.5 h-[1em] bg-current inline-block transition-opacity duration-75 ${
            showCursor ? 'opacity-100' : 'opacity-0'
          }`}
        />
      </div>
    </div>
  );
}