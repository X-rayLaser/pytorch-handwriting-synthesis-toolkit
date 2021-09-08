import { generateHandwriting } from './utils';

onmessage = function(e) {
    const text = e.data[0];
    const bias = e.data[1];

    const onProgress = (progress, results) => {
        self.postMessage({event: "progressChanged", value: progress, results: results});
    };

    generateHandwriting(text, bias, onProgress).then(results => {
        self.postMessage({event: "resultsReady", results: results});
    });
}

/*
export const startWorker = (text, bias) => {
    const onProgress = (progress, results) => {
        // eslint-disable-next-line
        self.postMessage({event: "progressChanged", value: progress, results: results});
    };
    generateHandwriting(text, bias, onProgress).then(results => {

        // eslint-disable-next-line
        self.postMessage({event: "resultsReady", results: results});
    });
};
*/