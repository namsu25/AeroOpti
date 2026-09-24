import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // maplibre-gl ships its render worker as a separate ESM chunk with a default
  // export; Vite's dependency pre-bundler mangles that interop and throws
  // "does not provide an export named 'default'" at runtime. Excluding it
  // from pre-bundling makes Vite serve it as native ESM instead, which works.
  optimizeDeps: {
    exclude: ["maplibre-gl"],
  },
})
