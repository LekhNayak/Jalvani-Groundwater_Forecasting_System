/** Colours are CSS variables (src/index.css) so light and dark themes swap in one place. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "var(--surface-1)", plane: "var(--page-plane)", ink: "var(--text-primary)",
        ink2: "var(--text-secondary)", muted: "var(--text-muted)", line: "var(--gridline)",
        water: "var(--series-1)", fall: "var(--fall)", good: "var(--status-good)",
        warn: "var(--status-warning)", serious: "var(--status-serious)", critical: "var(--status-critical)",
      },
      fontFamily: { sans: ['"Noto Sans"', '"Noto Sans Devanagari"', '"Noto Sans Gujarati"', "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
