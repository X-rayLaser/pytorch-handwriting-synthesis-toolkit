let worker = null;

onmessage = function(e) {
    if (e.data.event === "start") {
        if (worker === null) {
            worker = new Worker(new URL("./worker.js", import.meta.url));

            let workerListener = e => {
                self.postMessage(e.data);
            }
            worker.addEventListener('message', workerListener);
        }

        let [text, bias, primingSequence, primingText] = e.data.params;
        worker.postMessage([text, bias, primingSequence, primingText]);
    } else if (e.data.event === "abort") {
        if (worker === null) {
            //disregard
            return;
        }

        worker.terminate();
        worker = null;
        self.postMessage({event: "aborted"})
    }

}

onerror = function(e) {
    //propagate any error raised in spawned worker
    throw e;
}