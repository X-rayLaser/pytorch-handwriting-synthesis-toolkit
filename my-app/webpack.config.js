const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  target: ['web'],
  entry: './src/index.js',
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'build'),
    library: {
        type: 'umd'
    }
  },
  module: {
    rules: [
        {
            test: /\.m?js$/,
            exclude: /(node_modules|bower_components)/,
            use: {
                loader: 'babel-loader',
                options: {
                    presets: ['@babel/preset-env', '@babel/preset-react'
                    ]
                }
            }
        },
        {
            test: /\.css/,
            use: ['style-loader', 'css-loader']
        }
    ]
  },
  plugins: [new CopyPlugin({
      // Use copy plugin to copy *.wasm to output folder.
      patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
    }),
    new CopyPlugin({
      // Use copy plugin to copy *.onnx to output folder.
      patterns: [{ from: './*.onnx', to: '[name][ext]' }]
    }),
    new HtmlWebpackPlugin({
      filename: './index.html',
      template: './src/index_template.html'
    })
  ],
  mode: "production"
};
