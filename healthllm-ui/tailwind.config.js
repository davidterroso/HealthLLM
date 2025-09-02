module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        typing: {
          '0%': { width: '0ch' },
          '100%': { width: '10ch' },
        },
        blink: {
          '0%,100%': { 'border-color': 'transparent' },
          '50%': { 'border-color': 'currentColor' },
        },
      },
      animation: {
        typewriter: 'typing 2.5s steps(10, end) forwards, blink .75s step-end infinite',
      },
    }
  },
  plugins: [],
}