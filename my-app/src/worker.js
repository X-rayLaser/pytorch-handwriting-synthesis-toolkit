import { generateHandwriting, generateHandwritingWithPriming } from './utils';


self.addEventListener("unhandledrejection", function(event) { 
    throw event.reason;
});

onmessage = function(e) {
    const text = e.data[0];
    const bias = e.data[1];
    const primingSequence = e.data[2];
    const primingText = e.data[3];

    const onProgress = (progress, results) => {
        self.postMessage({event: "progressChanged", value: progress, results: results});
    };

    if (primingSequence.length > 0 && primingText.length > 0) {
        generateHandwritingWithPriming(primingSequence, primingText, text, bias, onProgress).then(results => {
            self.postMessage({event: "resultsReady", results: results});
        }).catch(reason => {
            throw `Error when generating handwriting with priming: ${reason}`;
        });
    } else {
        generateHandwriting(text, bias, onProgress).then(results => {
            self.postMessage({event: "resultsReady", results: results});
        }).catch(reason => {
            throw `Error when generating handwriting: ${reason}`;
        });
    }
}
