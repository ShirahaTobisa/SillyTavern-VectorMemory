// SillyTavern loads third-party extension entry points as classic scripts.
// Dynamic import keeps the implementation modular while remaining installable
// from an extension repository without importing ST internal modules.
let runtimeModule;
const runtimeReady = import('./src/runtime.js')
  .then(module => {
    runtimeModule = module;
    module.initVectorMemory();
    return module;
  })
  .catch(error => {
    console.error('[VectorMemory] failed to load extension', error);
    throw error;
  });

globalThis.vectorMemoryInterceptor = async (...args) => {
  const module = runtimeModule || await runtimeReady;
  return module.vectorMemoryInterceptor(...args);
};
