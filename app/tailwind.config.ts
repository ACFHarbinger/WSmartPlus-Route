import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: {
          bg: "#0f0f1a",
          surface: "#1a1a2e",
          elevated: "#20203a",
          border: "#2d2d50",
          hover: "#26264a",
          muted: "#9090b0",
        },
        accent: {
          primary: "#6366f1",
          secondary: "#818cf8",
          success: "#34d399",
          warning: "#fbbf24",
          danger: "#f87171",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
