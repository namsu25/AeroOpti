import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// No StrictMode: it double-invokes effects in dev, which creates and tears down
// a MapLibre GL map instance back-to-back and can leave the second instance's
// tile/worker state stuck (map.loaded() never resolves). MapLibre owns real
// WebGL/worker resources outside React's render model, so it doesn't tolerate
// that double-mount the way plain DOM effects do.
createRoot(document.getElementById('root')!).render(<App />)
