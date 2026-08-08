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
 *
 * IMPORTANTE — mantenimiento: el navegador solo revisa si hay una
 * versión nueva de ESTE archivo (service-worker.js) comparándolo byte
 * por byte con el anterior. Si solo cambias index.html (o el CSS/JS
 * dentro de él) y NO tocas este archivo, el navegador nunca se entera
 * de que hay algo nuevo y sigue sirviendo la versión vieja cacheada
 * indefinidamente. Regla práctica: cada vez que cambies cualquier
 * archivo del app shell (index.html, manifest.json, íconos), sube
 * también el número de CACHE_NAME de este archivo (v2, v3, v4...)
 * para forzar a que el navegador detecte el cambio y refresque la caché.
 */

const CACHE_NAME = "nfl-predictor-v2";
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
