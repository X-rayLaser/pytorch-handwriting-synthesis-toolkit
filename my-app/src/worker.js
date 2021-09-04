import { generateHandwriting } from './utils';

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
