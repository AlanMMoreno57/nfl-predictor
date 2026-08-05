/**
 * service-worker.js
 * ==================
 * Habilita que la PWA funcione instalada (ícono propio, sin barra de
 * navegador) y con acceso offline a la última versión vista.
 *
 * Estrategia:
 *  - App shell (index.html, manifest, íconos): cache-first — carga
 *    instantánea, se actualiza solo cuando cambia la versión CACHE_NAME.
 *  - predictions.json: network-first — siempre intenta traer los datos
 *    más recientes; si no hay conexión, cae al último JSON cacheado
 *    (así el usuario ve "algo" aunque esté sin internet, en vez de
 *    pantalla en blanco).
 */

const CACHE_NAME = "nfl-predictor-v1";
const APP_SHELL = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  if (url.pathname.endsWith("predictions.json")) {
    // Network-first para los datos: prioriza lo más fresco.
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Cache-first para el resto del app shell.
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
