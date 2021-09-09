import './App.css';
import React from 'react';
import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Accordion from 'react-bootstrap/Accordion';
import { SketchPicker } from 'react-color';


class CanvasDrawer {
  constructor(canvas, marginX, marginY, lineWidth, background, strokeColor) {
    this.wasEos = false;
    this.context = canvas.getContext('2d', { alpha: false});
    this.marginX = marginX;
    this.marginY = marginY;
    this.context.clearRect(0, 0, canvas.width, canvas.height);
    this.context.fillStyle = background;
    this.context.strokeStyle = strokeColor;
    if (background !== '#fff') {
      this.context.fillRect(0, 0, canvas.width, canvas.height);
    }
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


let worker = new Worker(new URL("./worker.js", import.meta.url));


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
      showBackgroundPicker: false,
      showStrokeColorPicker: false,
      background: '#fff',
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
    this.handleChangeBackground = this.handleChangeBackground.bind(this);
    this.handleChangeStrokeColor = this.handleChangeStrokeColor.bind(this);
    this.handleBiasChange = this.handleBiasChange.bind(this);
    this.adjustCanvasSize = this.adjustCanvasSize.bind(this);
    this.workerListener = null;
  }

  resetGeometry() {
    let defaultLogicalWidth = window.innerWidth;
    let defaultLogicalHeight = 1000;

    this.setState({
      scale: 1,
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
    
    this.workerListener = e => {
      if (e.data.event === "resultsReady") {
        this.handleCompletion(e);
      }

      if (e.data.event === "progressChanged") {
        this.handleProgress(e);
      }
    };


    worker.addEventListener('message', this.workerListener);
  }

  handleProgress(e) {
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
        progress: e.data.value,
        points: newPoints
      }
    });
  }

  handleCompletion(e) {
    this.setState({
      points:e.data.results,
      done: true,
      progress: 0
    });
  }

  componentDidUpdate() {
    this.updateCanvas();
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.adjustCanvasSize);
    if (this.workerListener) {
      worker.removeEventListener('message', this.workerListener);
    }
  }

  adjustCanvasSize() {
    this.forceUpdate();
  }

  updateCanvas() {
    const canvas = this.canvasRef.current;

    const marginX = this.state.geometry.minX;
    const marginY = this.state.geometry.minY;

    const minWidth = 5;
    let canvasWidth = window.innerWidth * this.state.scale;
    const lineWidth = Math.floor(this.state.geometry.width / canvasWidth) + minWidth;
    const backgroundColor = this.state.background;
    const strokeColor = this.state.strokeColor;
    const drawer = new CanvasDrawer(canvas, marginX, marginY, lineWidth, backgroundColor, strokeColor);

    drawer.draw(this.state.points);
    drawer.finish();
  }

  handleClick() {
    if (this.state.bias < 0) {
      window.alert("Negative bias is not allowed!");
      return;
    }

    if (this.state.text.length < 6) {
      window.alert("Text must contain at least 6 characters. Please, try again.");
      return;
    }

    if (this.state.text.length >= 50) {
      window.alert("Text must contain fewer thatn 50 characters. Please, try again.");
      return;
    }
    this.resetGeometry();
    this.setState({points: [], done: false});

    worker.postMessage([this.state.text, this.state.bias]);
  }

  handleZoomIn() {
    if (this.canZoomIn()) {
      this.setState((state, cb) => ({scale: state.scale * 2}));
    }
  }

  handleZoomOut() {
    if (this.canZoomOut()) {
      this.setState((state, cb) => ({scale: state.scale / 2}));
    }
  }

  canZoomIn() {
    return this.state.scale < 10;
  }

  canZoomOut() {
    return this.state.scale > 0.1;
  }

  isScaleWithinBounds() {
    return this.state.scale > 0.1 && this.state.scale < 10;
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
    try {
      let value = parseFloat(e.target.value);
      if (value >= 0 && value <= 10) {
        this.setState({bias: value});  
      }
    } catch (e) {
      console.error(e);
    }
  }

  handleChangeBackground(color) {
    this.setState({background: color.hex});
  }

  handleChangeStrokeColor(color) {
    this.setState({strokeColor: color.hex});
  }
  render() {
    let scaledWidth = window.innerWidth * this.state.scale;
    let canvasHeight = Math.round(scaledWidth / this.getAspectRatio());
    let canvasWidth = Math.round(scaledWidth);

    return (
      <div className="App">
        <Container>
          <textarea className="mb-2" placeholder="Enter text to generate a handwriting for" 
                    value={this.state.text} onChange={this.handleChange}>

          </textarea>
          <Accordion collapse className="mb-2">
            <Accordion.Item eventKey="0">
              <Accordion.Header>Settings</Accordion.Header>
              <Accordion.Body>
                <Form>
                  <Row className="align-items-center mb-2">

                    <Col xs="auto" style={{margin: 'auto'}}>
                      <Form.Label htmlFor="inlineFormInput">
                        Bias
                      </Form.Label>
                        <Form.Control type="number" id="inlineFormInput"
                                      value={this.state.bias} min={0} max={100} step={0.1} 
                                      onChange={this.handleBiasChange} />
                      </Col>
                  </Row>
                  <Row className="mb-2">

                  {!this.state.showBackgroundPicker &&
                    <Col>
                      <Button onClick={e => this.setState({showBackgroundPicker: true})}>
                        Choose background color
                      </Button>
                    </Col>
                  }
                  {this.state.showBackgroundPicker &&
                    <Col>
                      <div style={{width: 'auto'}}>
                        <SketchPicker
                          style={{margin:'auto'}}
                          color={ this.state.background }
                          onChangeComplete={ this.handleChangeBackground }
                        />
                      </div>
                      <Button onClick={e => this.setState({showBackgroundPicker: false})}>
                        Close color picker
                      </Button>
                    </Col>
                  }
                  </Row>
                  <Row>
                    {!this.state.showStrokeColorPicker &&
                      <Col>
                        <Button onClick={e => this.setState({showStrokeColorPicker: true})}>
                          Choose stroke color
                        </Button>
                      </Col>
                    }
                    {this.state.showStrokeColorPicker &&
                      <Col>
                        <div style={{width: 'auto'}}>
                          <SketchPicker
                            style={{margin:'auto'}}
                            color={ this.state.strokeColor }
                            onChangeComplete={ this.handleChangeStrokeColor }
                          />
                        </div>
                        <Button onClick={e => this.setState({showStrokeColorPicker: false})}>
                          Close color picker
                        </Button>
                      </Col>
                    }
                  </Row>
                </Form>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
          <Row className="mb-2">
            <Col>
              <Button onClick={this.handleClick} disabled={this.state.text.trim() === "" || !this.state.done}>
                Generate handwriting
              </Button>
            </Col>
          </Row>

          {!this.state.done && 
            <Row className="mb-2">
              <Col>
                <Row>
                  <Col>
                    <h4>Generating a handwriting, please wait...</h4>
                  </Col>
                </Row>
                <Row>
                  <Col>
                    <ProgressBar now={this.state.progress} />
                  </Col>
                </Row>
              </Col>
            </Row>
          }
          {this.state.done && this.state.points.length > 0 &&
          <Row className="mb-2">
            <Col>
              <ButtonGroup size="sm">
                <Button variant="secondary" onClick={this.handleZoomIn} disabled={!this.canZoomIn()}>
                  Zoom In
                </Button>
                <Button variant="secondary" onClick={this.handleZoomOut} disabled={!this.canZoomOut()}>
                  Zoom out
                </Button>
              </ButtonGroup>
            </Col>
          </Row>
          }
        </Container>
        <div style={{ overflow: 'auto'}}>
          <canvas ref={this.canvasRef} width={this.state.geometry.width} height={this.state.geometry.height} 
                  style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`}} ></canvas>
        </div>
      </div>
    );
  }
}


function App() {
  return (
    <div>
      <Container>
        <h4 style={{ textAlign: 'center'}}>This is a handwriting synthesis demo. It is a Javascript port of 
          <a href="https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit"> pytorch-handwriting-synthesis-toolkit</a> repository.
        </h4>
      </Container>
      <HandwritingScreen />
    </div>
  );
}

export default App;
