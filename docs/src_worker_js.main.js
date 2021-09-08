/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else {
		var a = factory();
		for(var i in a) (typeof exports === 'object' ? exports : root)[i] = a[i];
	}
})(self, function() {
return /******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/utils.js":
/*!**********************!*\
  !*** ./src/utils.js ***!
  \**********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"generateHandwriting\": () => (/* binding */ generateHandwriting)\n/* harmony export */ });\n/* harmony import */ var core_js_stable__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! core-js/stable */ \"./node_modules/core-js/stable/index.js\");\n/* harmony import */ var core_js_stable__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(core_js_stable__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var regenerator_runtime_runtime__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! regenerator-runtime/runtime */ \"./node_modules/regenerator-runtime/runtime.js\");\n/* harmony import */ var regenerator_runtime_runtime__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(regenerator_runtime_runtime__WEBPACK_IMPORTED_MODULE_1__);\n/* harmony import */ var multivariate_normal__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! multivariate-normal */ \"./node_modules/multivariate-normal/index.js\");\nfunction asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }\n\nfunction _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, \"next\", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, \"throw\", err); } _next(undefined); }); }; }\n\nfunction _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }\n\nfunction _nonIterableSpread() { throw new TypeError(\"Invalid attempt to spread non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.\"); }\n\nfunction _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === \"string\") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === \"Object\" && o.constructor) n = o.constructor.name; if (n === \"Map\" || n === \"Set\") return Array.from(o); if (n === \"Arguments\" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }\n\nfunction _iterableToArray(iter) { if (typeof Symbol !== \"undefined\" && iter[Symbol.iterator] != null || iter[\"@@iterator\"] != null) return Array.from(iter); }\n\nfunction _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }\n\nfunction _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }\n\nfunction _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError(\"Cannot call a class as a function\"); } }\n\nfunction _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if (\"value\" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }\n\nfunction _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }\n\n\n\n\nvar ort = __webpack_require__(/*! onnxruntime-web */ \"./node_modules/onnxruntime-web/dist/ort-web.min.js\");\n\n\n/*\nA hack to work around an issue with missing .wasm files when app is \ndeployed at Github Pages. onnxruntime-web tries fetching them relative to the / instead of\n[repository-name]/.\n */\n\nvar oldFetch = self.fetch;\n\nself.fetch = function () {\n  if (location.hostname === \"localhost\") {\n    return oldFetch.apply(this, arguments);\n  }\n\n  var oldUrl = arguments[0];\n\n  if (oldUrl.endsWith(\"ort-wasm.wasm\")) {\n    arguments[0] = \"https://x-raylaser.github.io/pytorch-handwriting-synthesis-toolkit/ort-wasm.wasm\";\n  }\n\n  return oldFetch.apply(this, arguments);\n};\n\nvar Tokenizer = /*#__PURE__*/function () {\n  function Tokenizer() {\n    _classCallCheck(this, Tokenizer);\n\n    this.charset = \" !\\\"#%'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[abcdefghijklmnopqrstuvwxyz\";\n    this.charToCode = new Map();\n    this.codeToChar = new Map();\n    var charset = this.charset;\n\n    for (var i = 0; i < charset.length; i++) {\n      var _char = charset[i];\n      this.charToCode.set(_char, i + 1);\n      this.codeToChar.set(i + 1, _char);\n    }\n  }\n\n  _createClass(Tokenizer, [{\n    key: \"tokenize\",\n    value: function tokenize(s) {\n      var tokens = [];\n      var charToCode = this.charToCode;\n\n      for (var i = 0; i < s.length; i++) {\n        var ch = s[i];\n\n        if (charToCode.has(ch)) {\n          tokens.push(charToCode.get(ch));\n        } else {\n          tokens.push(0);\n        }\n      }\n\n      return tokens;\n    }\n  }, {\n    key: \"detokenize\",\n    value: function detokenize(tokens) {\n      var s = '';\n\n      for (var i = 0; i < tokens.length; i++) {\n        var t = tokens[i];\n        s = s + this.codeToChar.get(t);\n      }\n\n      return s;\n    }\n  }, {\n    key: \"charsetSize\",\n    value: function charsetSize() {\n      return this.charset.length + 1;\n    }\n  }]);\n\n  return Tokenizer;\n}();\n\nfunction toOneHot(token, numClasses) {\n  var arr = [];\n\n  for (var i = 0; i < numClasses; i++) {\n    arr[i] = 0;\n  }\n\n  arr[token] = 1;\n  return arr;\n}\n\nfunction prepareString(s) {\n  var tokenizer = new Tokenizer();\n  var charsetSize = tokenizer.charsetSize();\n  var tokens = tokenizer.tokenize(s);\n  var arr = [];\n\n  for (var i = 0; i < tokens.length; i++) {\n    var oneHot = toOneHot(tokens[i], charsetSize);\n    arr.push.apply(arr, _toConsumableArray(oneHot));\n  }\n\n  return new ort.Tensor('float32', new Float32Array(arr), [1, s.length, charsetSize]);\n}\n\nfunction generateHandwriting(_x, _x2, _x3) {\n  return _generateHandwriting.apply(this, arguments);\n}\n\nfunction _generateHandwriting() {\n  _generateHandwriting = _asyncToGenerator( /*#__PURE__*/regeneratorRuntime.mark(function _callee(text, biasValue, onProgress) {\n    var session, normalizationParams, x, c, w, k, h1, c1, h2, c2, h3, c3, biasArray, bias, prevX, prevY, outputs, avgStepsPerLetter, steps, updateInterval, i, feeds, results, pi, mu, sd, eos, component, numComponents, xTemp, mu1, mu2, sd1, sd2, ro, sample, xDenormalized, yDenormalized, u, stepsComplete, currentBatch, fromIndex, toIndex;\n    return regeneratorRuntime.wrap(function _callee$(_context) {\n      while (1) {\n        switch (_context.prev = _context.next) {\n          case 0:\n            _context.next = 2;\n            return ort.InferenceSession.create('./synthesis_network_52.onnx');\n\n          case 2:\n            session = _context.sent;\n            normalizationParams = {\n              mu: [8.217162132263184, 0.1212363988161087, 0.0],\n              sd: [42.34926223754883, 37.07347869873047, 1.0]\n            };\n            x = new ort.Tensor('float32', new Float32Array(3), [1, 1, 3]);\n            c = prepareString(text);\n            w = new ort.Tensor('float32', new Float32Array(80), [1, 1, 80]);\n            k = new ort.Tensor('float32', new Float32Array(10), [1, 10]);\n            h1 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            c1 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            h2 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            c2 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            h3 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            c3 = new ort.Tensor('float32', new Float32Array(400), [1, 400]);\n            biasArray = new Float32Array(1);\n            biasArray[0] = biasValue;\n            bias = new ort.Tensor('float32', biasArray, [1]);\n            prevX = 0;\n            prevY = 0;\n            outputs = [];\n            avgStepsPerLetter = 50;\n            steps = avgStepsPerLetter * text.length;\n            updateInterval = 1;\n            i = 0;\n\n          case 24:\n            if (!(i < steps)) {\n              _context.next = 68;\n              break;\n            }\n\n            feeds = {\n              'x': x,\n              'c': c,\n              'w.1': w,\n              'k.1': k,\n              'h1.1': h1,\n              'c1.1': c1,\n              'h2.1': h2,\n              'c2.1': c2,\n              'h3.1': h3,\n              'c3.1': c3,\n              'bias': bias\n            };\n            _context.next = 28;\n            return session.run(feeds);\n\n          case 28:\n            results = _context.sent;\n            pi = results.pi.data;\n            mu = results.mu.data;\n            sd = results.sd.data;\n            eos = results.eos.data;\n            w = new ort.Tensor('float32', results.w.data, [1, 1, 80]);\n            k = new ort.Tensor('float32', results.k.data, [1, 10]);\n            h1 = new ort.Tensor('float32', results.h1.data, [1, 400]);\n            c1 = new ort.Tensor('float32', results.c1.data, [1, 400]);\n            h2 = new ort.Tensor('float32', results.h2.data, [1, 400]);\n            c2 = new ort.Tensor('float32', results.c2.data, [1, 400]);\n            h3 = new ort.Tensor('float32', results.h3.data, [1, 400]);\n            c3 = new ort.Tensor('float32', results.c3.data, [1, 400]);\n            component = randomChoice(pi);\n            numComponents = pi.length;\n            xTemp = new Float32Array(3);\n            mu1 = mu[component];\n            mu2 = mu[numComponents + component];\n            sd1 = sd[component];\n            sd2 = sd[numComponents + component];\n            ro = results.ro.data[component];\n            sample = sampleFromNormal(mu1, mu2, sd1, sd2, ro);\n            xTemp[0] = sample.x;\n            xTemp[1] = sample.y;\n\n            if (eos[0] > 0.5) {\n              xTemp[2] = 1.0;\n            } else {\n              xTemp[2] = 0.0;\n            }\n\n            xDenormalized = xTemp[0] * normalizationParams.sd[0] + normalizationParams.mu[0];\n            yDenormalized = xTemp[1] * normalizationParams.sd[1] + normalizationParams.mu[1];\n            prevX += xDenormalized;\n            prevY += yDenormalized;\n            u = text.length;\n\n            if (!is_end_of_string(results.phi.data, u, xTemp[2])) {\n              _context.next = 61;\n              break;\n            }\n\n            xTemp[2] = 1.0;\n            return _context.abrupt(\"break\", 68);\n\n          case 61:\n            outputs.push({\n              \"x\": Math.round(prevX),\n              \"y\": Math.round(prevY),\n              \"eos\": xTemp[2]\n            });\n            stepsComplete = i + 1;\n\n            if (stepsComplete % updateInterval === 0 && outputs.length > 0) {\n              currentBatch = void 0;\n              fromIndex = Math.max(0, stepsComplete - updateInterval);\n              toIndex = stepsComplete;\n              currentBatch = outputs.slice(fromIndex, toIndex);\n              onProgress(Math.round(i / steps * 100), currentBatch);\n            }\n\n            x = new ort.Tensor('float32', xTemp, [1, 1, 3]);\n\n          case 65:\n            i++;\n            _context.next = 24;\n            break;\n\n          case 68:\n            return _context.abrupt(\"return\", outputs);\n\n          case 69:\n          case \"end\":\n            return _context.stop();\n        }\n      }\n    }, _callee);\n  }));\n  return _generateHandwriting.apply(this, arguments);\n}\n\nfunction is_end_of_string(phi, string_length, is_eos) {\n  var last_phi = phi[string_length - 1];\n  return last_phi > 0.8 || argmax(phi) == string_length - 1 && is_eos;\n}\n\nfunction argmax(a) {\n  return a.indexOf(Math.max.apply(Math, _toConsumableArray(a)));\n}\n\nfunction sampleFromNormal(mu1, mu2, sd1, sd2, ro) {\n  var cov_x_y = ro * sd1 * sd2;\n  var sigma = [[Math.pow(sd1, 2), cov_x_y], [cov_x_y, Math.pow(sd2, 2)]];\n  var loc = [mu1, mu2];\n  var gmm = (0,multivariate_normal__WEBPACK_IMPORTED_MODULE_2__[\"default\"])(loc, sigma);\n  var v = gmm.sample();\n  return {\n    x: v[0],\n    y: v[1]\n  };\n}\n\nfunction randomChoice(probs) {\n  return makeChoice(0, probs);\n}\n\nfunction makeChoice(currentIndex, probs) {\n  if (probs.length === 1) {\n    return currentIndex;\n  }\n\n  if (Math.random() < probs[currentIndex]) {\n    return currentIndex;\n  }\n\n  var s = 0;\n\n  for (var i = currentIndex + 1; i < probs.length; i++) {\n    s += probs[i];\n  }\n\n  for (var i = currentIndex + 1; i < probs.length; i++) {\n    probs[i] = probs[i] / s;\n  }\n\n  return makeChoice(currentIndex + 1, probs);\n}\n\n//# sourceURL=webpack://handwriting_demo/./src/utils.js?");

/***/ }),

/***/ "./src/worker.js":
/*!***********************!*\
  !*** ./src/worker.js ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils */ \"./src/utils.js\");\n\n\nonmessage = function onmessage(e) {\n  var text = e.data[0];\n  var bias = e.data[1];\n\n  var onProgress = function onProgress(progress, results) {\n    self.postMessage({\n      event: \"progressChanged\",\n      value: progress,\n      results: results\n    });\n  };\n\n  (0,_utils__WEBPACK_IMPORTED_MODULE_0__.generateHandwriting)(text, bias, onProgress).then(function (results) {\n    self.postMessage({\n      event: \"resultsReady\",\n      results: results\n    });\n  });\n};\n/*\nexport const startWorker = (text, bias) => {\n    const onProgress = (progress, results) => {\n        // eslint-disable-next-line\n        self.postMessage({event: \"progressChanged\", value: progress, results: results});\n    };\n    generateHandwriting(text, bias, onProgress).then(results => {\n\n        // eslint-disable-next-line\n        self.postMessage({event: \"resultsReady\", results: results});\n    });\n};\n*/\n\n//# sourceURL=webpack://handwriting_demo/./src/worker.js?");

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			id: moduleId,
/******/ 			loaded: false,
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Flag the module as loaded
/******/ 		module.loaded = true;
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = __webpack_modules__;
/******/ 	
/******/ 	// the startup function
/******/ 	__webpack_require__.x = () => {
/******/ 		// Load entry module and return exports
/******/ 		// This entry module depends on other loaded chunks and execution need to be delayed
/******/ 		var __webpack_exports__ = __webpack_require__.O(undefined, ["vendors-node_modules_core-js_stable_index_js-node_modules_multivariate-normal_index_js-node_m-3904ba"], () => (__webpack_require__("./src/worker.js")))
/******/ 		__webpack_exports__ = __webpack_require__.O(__webpack_exports__);
/******/ 		return __webpack_exports__;
/******/ 	};
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/chunk loaded */
/******/ 	(() => {
/******/ 		var deferred = [];
/******/ 		__webpack_require__.O = (result, chunkIds, fn, priority) => {
/******/ 			if(chunkIds) {
/******/ 				priority = priority || 0;
/******/ 				for(var i = deferred.length; i > 0 && deferred[i - 1][2] > priority; i--) deferred[i] = deferred[i - 1];
/******/ 				deferred[i] = [chunkIds, fn, priority];
/******/ 				return;
/******/ 			}
/******/ 			var notFulfilled = Infinity;
/******/ 			for (var i = 0; i < deferred.length; i++) {
/******/ 				var [chunkIds, fn, priority] = deferred[i];
/******/ 				var fulfilled = true;
/******/ 				for (var j = 0; j < chunkIds.length; j++) {
/******/ 					if ((priority & 1 === 0 || notFulfilled >= priority) && Object.keys(__webpack_require__.O).every((key) => (__webpack_require__.O[key](chunkIds[j])))) {
/******/ 						chunkIds.splice(j--, 1);
/******/ 					} else {
/******/ 						fulfilled = false;
/******/ 						if(priority < notFulfilled) notFulfilled = priority;
/******/ 					}
/******/ 				}
/******/ 				if(fulfilled) {
/******/ 					deferred.splice(i--, 1)
/******/ 					var r = fn();
/******/ 					if (r !== undefined) result = r;
/******/ 				}
/******/ 			}
/******/ 			return result;
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/compat get default export */
/******/ 	(() => {
/******/ 		// getDefaultExport function for compatibility with non-harmony modules
/******/ 		__webpack_require__.n = (module) => {
/******/ 			var getter = module && module.__esModule ?
/******/ 				() => (module['default']) :
/******/ 				() => (module);
/******/ 			__webpack_require__.d(getter, { a: getter });
/******/ 			return getter;
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/define property getters */
/******/ 	(() => {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = (exports, definition) => {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/ensure chunk */
/******/ 	(() => {
/******/ 		__webpack_require__.f = {};
/******/ 		// This file contains only the entry chunk.
/******/ 		// The chunk loading function for additional chunks
/******/ 		__webpack_require__.e = (chunkId) => {
/******/ 			return Promise.all(Object.keys(__webpack_require__.f).reduce((promises, key) => {
/******/ 				__webpack_require__.f[key](chunkId, promises);
/******/ 				return promises;
/******/ 			}, []));
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/get javascript chunk filename */
/******/ 	(() => {
/******/ 		// This function allow to reference async chunks and sibling chunks for the entrypoint
/******/ 		__webpack_require__.u = (chunkId) => {
/******/ 			// return url for filenames based on template
/******/ 			return "" + chunkId + ".main.js";
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/global */
/******/ 	(() => {
/******/ 		__webpack_require__.g = (function() {
/******/ 			if (typeof globalThis === 'object') return globalThis;
/******/ 			try {
/******/ 				return this || new Function('return this')();
/******/ 			} catch (e) {
/******/ 				if (typeof window === 'object') return window;
/******/ 			}
/******/ 		})();
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	(() => {
/******/ 		__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/node module decorator */
/******/ 	(() => {
/******/ 		__webpack_require__.nmd = (module) => {
/******/ 			module.paths = [];
/******/ 			if (!module.children) module.children = [];
/******/ 			return module;
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/publicPath */
/******/ 	(() => {
/******/ 		var scriptUrl;
/******/ 		if (__webpack_require__.g.importScripts) scriptUrl = __webpack_require__.g.location + "";
/******/ 		var document = __webpack_require__.g.document;
/******/ 		if (!scriptUrl && document) {
/******/ 			if (document.currentScript)
/******/ 				scriptUrl = document.currentScript.src
/******/ 			if (!scriptUrl) {
/******/ 				var scripts = document.getElementsByTagName("script");
/******/ 				if(scripts.length) scriptUrl = scripts[scripts.length - 1].src
/******/ 			}
/******/ 		}
/******/ 		// When supporting browsers where an automatic publicPath is not supported you must specify an output.publicPath manually via configuration
/******/ 		// or pass an empty string ("") and set the __webpack_public_path__ variable from your code to use your own logic.
/******/ 		if (!scriptUrl) throw new Error("Automatic publicPath is not supported in this browser");
/******/ 		scriptUrl = scriptUrl.replace(/#.*$/, "").replace(/\?.*$/, "").replace(/\/[^\/]+$/, "/");
/******/ 		__webpack_require__.p = scriptUrl;
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/importScripts chunk loading */
/******/ 	(() => {
/******/ 		// no baseURI
/******/ 		
/******/ 		// object to store loaded chunks
/******/ 		// "1" means "already loaded"
/******/ 		var installedChunks = {
/******/ 			"src_worker_js": 1
/******/ 		};
/******/ 		
/******/ 		// importScripts chunk loading
/******/ 		var installChunk = (data) => {
/******/ 			var [chunkIds, moreModules, runtime] = data;
/******/ 			for(var moduleId in moreModules) {
/******/ 				if(__webpack_require__.o(moreModules, moduleId)) {
/******/ 					__webpack_require__.m[moduleId] = moreModules[moduleId];
/******/ 				}
/******/ 			}
/******/ 			if(runtime) runtime(__webpack_require__);
/******/ 			while(chunkIds.length)
/******/ 				installedChunks[chunkIds.pop()] = 1;
/******/ 			parentChunkLoadingFunction(data);
/******/ 		};
/******/ 		__webpack_require__.f.i = (chunkId, promises) => {
/******/ 			// "1" is the signal for "already loaded"
/******/ 			if(!installedChunks[chunkId]) {
/******/ 				if(true) { // all chunks have JS
/******/ 					importScripts(__webpack_require__.p + __webpack_require__.u(chunkId));
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 		
/******/ 		var chunkLoadingGlobal = self["webpackChunkhandwriting_demo"] = self["webpackChunkhandwriting_demo"] || [];
/******/ 		var parentChunkLoadingFunction = chunkLoadingGlobal.push.bind(chunkLoadingGlobal);
/******/ 		chunkLoadingGlobal.push = installChunk;
/******/ 		
/******/ 		// no HMR
/******/ 		
/******/ 		// no HMR manifest
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/startup chunk dependencies */
/******/ 	(() => {
/******/ 		var next = __webpack_require__.x;
/******/ 		__webpack_require__.x = () => {
/******/ 			return __webpack_require__.e("vendors-node_modules_core-js_stable_index_js-node_modules_multivariate-normal_index_js-node_m-3904ba").then(next);
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
/******/ 	
/******/ 	// run startup
/******/ 	var __webpack_exports__ = __webpack_require__.x();
/******/ 	
/******/ 	return __webpack_exports__;
/******/ })()
;
});