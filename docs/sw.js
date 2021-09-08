self.addEventListener('fetch', function(event) {
    event.respondWith(async function() {
        if (event.request.url === "https://x-raylaser.github.io/ort-wasm.wasm") {
            return fetch("https://x-raylaser.github.io/pytorch-handwriting-synthesis-toolkit/ort-wasm.wasm");
        }
        return fetch(event.request.url);
    }());
});