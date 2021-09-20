import "core-js/stable";
import "regenerator-runtime/runtime";

const ort = require('onnxruntime-web');
import MultivariateNormal from "multivariate-normal";

/*
A hack to work around an issue with missing .wasm files when app is 
deployed at Github Pages. onnxruntime-web tries fetching them relative to the / instead of
[repository-name]/.
*/
const oldFetch = self.fetch;
self.fetch = function() {
  if (location.hostname === "localhost") {
    return oldFetch.apply(this, arguments);
  }

  let oldUrl = arguments[0];

  if (oldUrl.endsWith("ort-wasm.wasm")) {
    arguments[0] = "https://x-raylaser.github.io/pytorch-handwriting-synthesis-toolkit/ort-wasm.wasm";
  }
  
  return oldFetch.apply(this, arguments);
}


class Tokenizer {
    constructor() {
        this.charset = " !\"#%'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[abcdefghijklmnopqrstuvwxyz";
        this.charToCode = new Map();
        this.codeToChar = new Map();

        const charset = this.charset;
        for (let i = 0; i < charset.length; i++) {
            const char = charset[i];
            this.charToCode.set(char, i + 1);
            this.codeToChar.set(i + 1, char);
        }
    }

    tokenize(s) {
        let tokens = [];
        const charToCode = this.charToCode;

        for (let i = 0; i < s.length; i++) {
            const ch = s[i];

            if (charToCode.has(ch)) {
                tokens.push(charToCode.get(ch));
            } else {
                tokens.push(0);
            }
        }

        return tokens;
    }

    detokenize(tokens) {
        let s = '';
        for (let i = 0; i<tokens.length; i++) {
          const t = tokens[i];
          s = s + this.codeToChar.get(t);
        }

        return s;
    }

    charsetSize() {
        return this.charset.length + 1;
    }
}


  
function toOneHot(token, numClasses) {
    let arr = [];

    for (let i = 0; i < numClasses; i++) {
      arr[i] = 0;
    }

    arr[token] = 1
    return arr;
}


function prepareString(s) {
    const tokenizer = new Tokenizer();
  
    const charsetSize = tokenizer.charsetSize();

    const tokens = tokenizer.tokenize(s);

    const arr = [];
    for (let i = 0; i < tokens.length; i++) {
      const oneHot = toOneHot(tokens[i], charsetSize);
      arr.push(...oneHot);
    }

    return new ort.Tensor('float32', new Float32Array(arr), [1, s.length, charsetSize]);
}


export async function generateHandwritingWithPriming(primingSequence, primingText, text, biasValue, onProgress) {
  const session = await ort.InferenceSession.create('./synthesis_network_52.onnx');

  const normalizationParams = {
    mu: [8.217162132263184, 0.1212363988161087, 0.0],
    sd: [42.34926223754883, 37.07347869873047, 1.0]
  };

  const biasArray = new Float32Array(1);
  biasArray[0] = biasValue;
  let bias = new ort.Tensor('float32', biasArray, [1]);

  let c = prepareString(primingText + text);
  let xPrimed = absoluteToOffsets(primingSequence);
  let xPrimedNormalized = [{x: 0, y: 0, eos: 0}].concat(normalize(xPrimed, normalizationParams.mu, normalizationParams.sd));

  let priming_steps = xPrimedNormalized.length;

  let feeds = getInitialInputs(c, bias);

  for (let t = 0; t < priming_steps - 1; t++ ) {
    let v = xPrimedNormalized[t];
    let x = new ort.Tensor('float32', new Float32Array([v.x, v.y, v.eos]), [1, 1, 3]);
    feeds.x = x;
    const results = await session.run(feeds);

    let w = new ort.Tensor('float32', results.w.data, [1, 1, 80]);

    let k = new ort.Tensor('float32', results.k.data, [1, 10]);

    let h1 = new ort.Tensor('float32', results.h1.data, [1, 400]);

    let c1 = new ort.Tensor('float32', results.c1.data, [1, 400]);

    let h2 = new ort.Tensor('float32', results.h2.data, [1, 400]);

    let c2 = new ort.Tensor('float32', results.c2.data, [1, 400]);

    let h3 = new ort.Tensor('float32', results.h3.data, [1, 400]);

    let c3 = new ort.Tensor('float32', results.c3.data, [1, 400]);

    feeds = {
      'c': c,
      'w.1': w,
      'k.1': k,
      'h1.1': h1,
      'c1.1': c1,
      'h2.1': h2,
      'c2.1': c2,
      'h3.1': h3,
      'c3.1': c3,
      'bias': bias
    };
  }

  let v = xPrimedNormalized[priming_steps - 1];
  let x = new ort.Tensor('float32', new Float32Array([v.x, v.y, v.eos]), [1, 1, 3]);
  feeds.x = x;
  const updateInterval = 1;
  const steps = estimateNumberOfSteps(text);
  return generate(feeds, normalizationParams, bias, onProgress, updateInterval, steps);
}


function absoluteToOffsets(points) {
  let res = [];
  let tail = points.slice(1);
  
  let prev = points[0];
  tail.forEach(p => {
    res.push({x: p.x - prev.x, y: p.y - prev.y, eos: p.eos});
    prev = p;
  });
  return res;
}


function normalize(points, mu, sd) {
  return points.map(p => ({
    x: (p.x - mu[0]) / sd[0],
    y: (p.y - mu[1]) / sd[1],
    eos: Number(p.eos)
  }));
}


export async function generateHandwriting(text, biasValue, onProgress) {
  const normalizationParams = {
    mu: [8.217162132263184, 0.1212363988161087, 0.0],
    sd: [42.34926223754883, 37.07347869873047, 1.0]
  };

  const steps = estimateNumberOfSteps(text);
  const updateInterval = 1;

  let c = prepareString(text);
  const biasArray = new Float32Array(1);
  biasArray[0] = biasValue;
  let bias = new ort.Tensor('float32', biasArray, [1]);

  let feeds = getInitialInputs(c, bias);

  return generate(feeds, normalizationParams, bias, onProgress, updateInterval, steps);
}

async function generate(initialFeeds, normalizationParams, bias, onProgress, updateInterval=1, steps=1000) {
  const session = await ort.InferenceSession.create('./synthesis_network_52.onnx');

  let prevX = 0;
  let prevY = 0;

  const outputs = [];

  let feeds = initialFeeds;

  let c = initialFeeds.c;

  for (var i = 0; i < steps; i++) {
    const results = await session.run(feeds);

    const pi = results.pi.data;
    const mu = results.mu.data;
    const sd = results.sd.data;
    const eos = results.eos.data;

    let w = new ort.Tensor('float32', results.w.data, [1, 1, 80]);

    let k = new ort.Tensor('float32', results.k.data, [1, 10]);

    let h1 = new ort.Tensor('float32', results.h1.data, [1, 400]);

    let c1 = new ort.Tensor('float32', results.c1.data, [1, 400]);

    let h2 = new ort.Tensor('float32', results.h2.data, [1, 400]);

    let c2 = new ort.Tensor('float32', results.c2.data, [1, 400]);

    let h3 = new ort.Tensor('float32', results.h3.data, [1, 400]);

    let c3 = new ort.Tensor('float32', results.c3.data, [1, 400]);

    const xTemp = sampleNextPoint(pi, mu, sd, results.ro.data, eos);

    const xDenormalized = xTemp[0] * normalizationParams.sd[0] + normalizationParams.mu[0];
    const yDenormalized = xTemp[1] * normalizationParams.sd[1] + normalizationParams.mu[1];
    
    prevX += xDenormalized;
    prevY += yDenormalized;

    let u = c.dims[1];

    if (u >= 6 && is_end_of_string(results.phi.data, u, xTemp[2])) {
      xTemp[2] = 1.0;
      break;
    }

    outputs.push({
        "x": Math.round(prevX),
        "y": Math.round(prevY),
        "eos": xTemp[2]
    });

    let stepsComplete = i + 1;
    if (stepsComplete % updateInterval === 0 && outputs.length > 0) {
      let currentBatch;
      let fromIndex = Math.max(0, stepsComplete - updateInterval);
      let toIndex = stepsComplete;
      currentBatch = outputs.slice(fromIndex, toIndex);
      onProgress(Math.round(i / steps * 100), currentBatch);
    }

    let x = new ort.Tensor('float32', xTemp, [1, 1, 3]);

    feeds = {
      'x': x,
      'c': c,
      'w.1': w,
      'k.1': k,
      'h1.1': h1,
      'c1.1': c1,
      'h2.1': h2,
      'c2.1': c2,
      'h3.1': h3,
      'c3.1': c3,
      'bias': bias
    };
  }

  return outputs;
}

function estimateNumberOfSteps(text, avgStepsPerLetter=35) {
  return avgStepsPerLetter * text.length;
}

function sampleNextPoint(pi, mu, sd, ro, eos) {
  const component = randomChoice(pi);
  const numComponents = pi.length;

  const xTemp = new Float32Array(3);
  const mu1 = mu[component];
  const mu2 = mu[numComponents + component];

  const sd1 = sd[component];
  const sd2 = sd[numComponents + component];
  const componentRo = ro[component];

  const sample = sampleFromNormal(mu1, mu2, sd1, sd2, componentRo);
  xTemp[0] = sample.x;
  xTemp[1] = sample.y;

  if (eos[0] > 0.5) {
    xTemp[2] = 1.0;
  } else {
    xTemp[2] = 0.0;
  }

  return xTemp;
}

function getInitialInputs(c, bias) {
  let x = new ort.Tensor('float32', new Float32Array(3), [1, 1, 3]);

  let w = new ort.Tensor('float32', new Float32Array(80), [1, 1, 80]);

  let k = new ort.Tensor('float32', new Float32Array(10), [1, 10]);

  let h1 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  let c1 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  let h2 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  let c2 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  let h3 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  let c3 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);

  return {
    'x': x,
    'c': c,
    'w.1': w,
    'k.1': k,
    'h1.1': h1,
    'c1.1': c1,
    'h2.1': h2,
    'c2.1': c2,
    'h3.1': h3,
    'c3.1': c3,
    'bias': bias
  };
}

function is_end_of_string(phi, string_length, is_eos) {
  const last_phi = phi[string_length - 1];
  return last_phi > 0.8 || (argmax(phi) == string_length - 1 && is_eos);
}

function argmax(a) {
  return a.indexOf(Math.max(...a));
}

function sampleFromNormal(mu1, mu2, sd1, sd2, ro) {
  const cov_x_y = ro * sd1 * sd2;
  const sigma = [[sd1 ** 2, cov_x_y], [cov_x_y, sd2 ** 2]];

  const loc = [mu1, mu2];
  const gmm =  MultivariateNormal(loc, sigma);

  const v = gmm.sample();

  return {x:v[0], y:v[1]};
}

function randomChoice(probs) {
  return makeChoice(0, probs);
}

function makeChoice(currentIndex, probs) {
    if (probs.length === 1) {
      return currentIndex;
    }
    
    if (Math.random() < probs[currentIndex]) {
      return currentIndex;
    }

    let s = 0;
    for (var i = currentIndex + 1; i < probs.length; i++) {
      s += probs[i];
    }

    for (var i = currentIndex + 1; i < probs.length; i++) {
      probs[i] = probs[i] / s;
    }

    return makeChoice(currentIndex + 1, probs);      
}