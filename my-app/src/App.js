import './App.css';
import React from 'react';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';
//const ort = require('onnxruntime-web');
import worker from 'workerize-loader!./worker';// eslint-disable-line import/no-webpack-loader-syntax


class CanvasDrawer {
  constructor(canvas, marginX, marginY, lineWidth) {
    this.wasEos = false;
    this.context = canvas.getContext('2d', { alpha: false});
    this.marginX = marginX;
    this.marginY = marginY;
    this.context.clearRect(0, 0, canvas.width, canvas.height);
    this.context.lineWidth = lineWidth || 30;
    this.context.beginPath();
  }

  draw(points) {
    const marginX = this.marginX;
    const marginY = this.marginY;
    const ctx = this.context;

    points.forEach(p => {
      if (this.wasEos) {
        ctx.moveTo(p.x - marginX, p.y - marginY);
        this.wasEos = false;
      } else {
        ctx.lineTo(p.x - marginX, p.y - marginY);
      }

      if (p.eos == 1) {
        this.wasEos = true;
      }
    });
  }

  finish() {
    this.context.stroke();
  }
}


class VirtualSurface {
  //the idea is to scale down all coordinates and then calculate the canvas size which will be smaller
  //thus performance should go up
  constructor(scaleFactor) {
    this.points = [];
    this.scale = scaleFactor;
  }
  
  push(points) {
    this.points.push(...points);
  }

  calculateGeometry() {
    let points = this.points.map(p => p / this.scale);
    calculateGeometry(points);
  }
}


function calculateGeometry(points) {
  let minX = 100000;
  let minY = 100000;
  let maxX = -10000;
  let maxY = -10000;

  points.forEach(p => {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);

    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  });

  let width = maxX - minX;
  let height = maxY - minY;
  return {
    minX,
    minY,
    width,
    height
  };
}


class HandwritingScreen extends React.Component {
  constructor(props) {
    super(props);
    this.canvasRef= React.createRef();

    let defaultLogicalWidth = window.innerWidth;
    let defaultLogicalHeight = 1000;

    this.state = {
      points: [],
      text: "",
      done: true,
      bias: 0.5,
      canvasHeight: window.innerWidth / (defaultLogicalWidth / defaultLogicalHeight),
      canvasWidth: defaultLogicalWidth,
      geometry: {
        x: 200,
        y: 200,
        width: defaultLogicalWidth,
        height: defaultLogicalHeight
      }
    };

    this.context = null;
    this.handleClick = this.handleClick.bind(this);
    this.handleZoomIn = this.handleZoomIn.bind(this);
    this.handleZoomOut = this.handleZoomOut.bind(this);
    this.handleCancel = this.handleCancel.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.handleBiasChange = this.handleBiasChange.bind(this);
    this.adjustCanvasSize = this.adjustCanvasSize.bind(this);
    this.worker = null;
  }

  resetGeometry() {
    let defaultLogicalWidth = window.innerWidth;
    let defaultLogicalHeight = 1000;

    this.setState({
      canvasHeight: window.innerWidth / (defaultLogicalWidth / defaultLogicalHeight),
      canvasWidth: defaultLogicalWidth,
      geometry: {
        x: 200,
        y: 200,
        width: defaultLogicalWidth,
        height: defaultLogicalHeight
      }
    });
  }

  getAspectRatio() {
    return this.state.geometry.width / this.state.geometry.height;
  }

  componentDidMount() {
    window.addEventListener('resize', this.adjustCanvasSize);

    this.context = this.canvasRef.current.getContext('2d');
    this.worker = worker();
    
    this.worker.addEventListener('message', e => {
      
      if (e.data.event === "resultsReady") {
        
        this.setState({
          points:e.data.results,
          done: true,
          progress: 0
        });
        
      }

      if (e.data.event === "progressChanged") {
        this.setState((state, cb) => {
          let newPoints = [...state.points, ...e.data.results];
          let newGeo = calculateGeometry(newPoints);
          const maxWidth = 10000;
          const maxHeight = 2000;
          newGeo.width = Math.max(window.innerWidth, state.geometry.width, newGeo.width);
          newGeo.height = Math.max(window.innerHeight, state.geometry.height, newGeo.height);

          newGeo.width = Math.min(newGeo.width, maxWidth);
          newGeo.height = Math.min(newGeo.height, maxHeight);

          if (newGeo.width > state.geometry.width) {
            const extraWidth = Math.round(state.geometry.width / 2);
            newGeo.width = state.geometry.width + extraWidth;
          }
          return {
            geometry: newGeo,
            canvasWidth: window.innerWidth,
            canvasHeight: window.innerWidth / (newGeo.width / newGeo.height),
            progress: e.data.value,
            points: newPoints
          }
        });
        
      }
    });
    
  }

  componentDidUpdate() {
    this.updateCanvas();
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.adjustCanvasSize);
  }

  adjustCanvasSize() {
      this.setState({
        canvasWidth: window.innerWidth,
        canvasHeight: window.innerWidth / this.getAspectRatio()
      });
  }

  updateCanvas() {
    const canvas = this.canvasRef.current;

    const marginX = this.state.geometry.minX;
    const marginY = this.state.geometry.minY;

    const minWidth = 5;
    const lineWidth = Math.floor(this.state.geometry.width / this.state.canvasWidth) + minWidth;
    const drawer = new CanvasDrawer(canvas, marginX, marginY, lineWidth);

    drawer.draw(this.state.points);
    drawer.finish();
  }

  handleClick() {
    this.resetGeometry();
    this.setState({points: [], done: false});
    this.worker.startWorker(this.state.text, this.state.bias);
  }

  handleZoomIn() {
    this.setState((state, cb) => ({
      canvasWidth: state.canvasWidth * 2,
      canvasHeight: state.canvasHeight * 2
    }));
  }

  handleZoomOut() {
    this.setState((state, cb) => ({
      canvasWidth: Math.round(state.canvasWidth / 2),
      canvasHeight: Math.round(state.canvasHeight / 2)
    }));
  }

  handleCancel() {
    if (this.worker) {
      this.worker.terminate();
      this.setState({
        points: [],
        done: true,
        progress: 0
      });
    }
  }

  handleChange(e) {
    this.setState({text: e.target.value});
  }
  handleBiasChange(e) {
    this.setState({bias: e.target.value});
  }
  render() {
    return (
      <div className="App">
        <textarea placeholder="Enter text for a handwriting" value={this.state.text} onChange={this.handleChange}>

        </textarea>
        <details>
          <summary>Settings</summary>
            <label>Bias</label>
            <input type="number" value={this.state.bias} min={0} max={100} step={0.1} onChange={this.handleBiasChange} />
        </details>
        <div>
          <Button onClick={this.handleClick} disabled={this.state.text.trim() === "" || !this.state.done}>
            Generate
          </Button>
        </div>

        {!this.state.done && <div>Generating a handwriting, please wait...</div>}
        {!this.state.done && <ProgressBar now={this.state.progress} />}
        {this.state.done && this.state.points.length > 0 &&
        <div>
          <Button onClick={this.handleZoomIn}>Zoom In</Button>
          <Button onClick={this.handleZoomOut}>Zoom out</Button>
        </div>
        }
        <div style={{ overflow: 'auto'}}>
          <canvas ref={this.canvasRef} width={this.state.geometry.width} height={this.state.geometry.height} 
                  style={{ width: `${this.state.canvasWidth}px`, height: `${this.state.canvasHeight}px`}} ></canvas>
        </div>
      </div>
    );
  }
}


function App() {
  return (
    <HandwritingScreen />
  );
}

export default App;
