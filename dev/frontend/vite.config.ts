import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Local development: the React app on :5173, the FastAPI backend on :8000 (proxied under /api).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", sourcemap: false },
  test: { environment: "node" },
});
