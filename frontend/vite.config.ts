import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Dev server on 5173 — the port `API_CORS_ORIGINS` allows by default.
 *
 * `/api` is proxied to the service rather than called cross-origin, so the
 * dev build and the built bundle served by `just serve` fetch exactly the same
 * relative URLs. One code path, and a misconfigured CORS origin cannot break
 * development in a way that only shows up in the browser console.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.API_ORIGIN ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
