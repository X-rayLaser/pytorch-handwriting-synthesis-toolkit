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
import CanvasDrawer from './drawing';
import PrimingModal from './PrimingModal';
import ScalableCanvas from './ScalableCanvas';


class CanvasConfiguration {
  constructor(backgroundColor='#fff', strokeColor='#000') {
    this.backgroundColor = backgroundColor;
    this.strokeColor = strokeColor;
    this.listener = null;
  }

  update(backgroundColor, strokeColor) {
    this.backgroundColor = backgroundColor;
    this.strokeColor = strokeColor;
    this.listener(this);
  }

  setListener(listener) {
    this.listener = listener;
  }
}


class HandwritingGenerationExecutor {
  constructor(worker) {
    this.subscribers = new Map();

    let workerListener = e => {
      if (e.data.event === "resultsReady") {
        this.notify("resultsReady", e.data.results);
      }

      if (e.data.event === "progressChanged") {
        this.notify("progressChanged", e.data.results, e.data.value);
      }
    };

    worker.addEventListener('message', workerListener);
  }

  runNewJob(text, bias, primingSequence, primingText) {
    this.notify("start");
    worker.postMessage([text, bias, primingSequence, primingText]);
  }

  notify(eventType, ...args) {
    if (!this.subscribers.has(eventType)) {
      return;
    }

    for (let subscriber of this.subscribers.get(eventType)) {
      subscriber(...args);
    }
  }

  abort() {
    //do nothing for now
    //todo:redesign worker to spawn its own workers
  }

  subscribe(eventType, listener) {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, []);
    }

    let arr = this.subscribers.get(eventType);
    arr.push(listener);
  }

  unsubscribe(eventType, listener) {
    if (!this.subscribers.has(eventType)) {
      return;
    }

    let eventSubscribers = this.subscribers.get(eventType);
    let index = eventSubscribers.indexOf(listener);
    if (index === -1) {
      console.error("Listener not found!!!");
      return;
    }

    eventSubscribers.splice(index, 1);
  }
}


let worker = new Worker(new URL("./worker.js", import.meta.url));
let canvasConfig = new CanvasConfiguration();
let executor = new HandwritingGenerationExecutor(worker);


export default function App() {
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


class VirtualCanvas {
  //the idea is to scale down all coordinates and then calculate the canvas size which will be smaller
  //thus performance should go up
  constructor(onArrival, onGeometryChange, scaleFactor=1) {
    this.buffer = [];
    this.lastChunk = [];
    this.scale = scaleFactor;
    this.minX = 0;
    this.minY = 0;
    this.maxX = 0;
    this.maxY = 0;

    this.onArrival = onArrival;
    this.onGeometryChange = onGeometryChange;

    this.MAX_WIDTH = 10000;
    this.MAX_HEIGHT = 4000;
  }
  
  addChunk(points) {
    let scaled = points.map(p => ({x: p.x * this.scale, y: p.y * this.scale, eos: p.eos}));
    let scaledX = scaled.map(p => p.x);
    let scaledY = scaled.map(p => p.y);

    let newMinX = Math.min(this.minX, ...scaledX);
    let newMinY = Math.min(this.minY, ...scaledY);

    let newMaxX = Math.max(this.maxX, ...scaledX);
    let newMaxY = Math.max(this.maxY, ...scaledY);

    if (newMinX !== this.minX || newMinY !== this.minY || newMaxX !== this.maxX || newMaxY !== this.maxY) {
      if (this.getWidth() > this.MAX_WIDTH || this.getHeight() > this.MAX_HEIGHT) {
        throw "Exceeded maximum size";
      }
      this.onGeometryChange(this);
    }

    this.minX = newMinX;
    this.minY = newMinY;
    this.maxX = newMaxX;
    this.maxY = newMaxY;

    this.lastChunk = scaled;
    this.buffer.push(...scaled);

    this.onArrival(this);
  }

  getWidth() {
    return this.maxX - this.minX; 
  }

  getHeight() {
    return this.maxY - this.minY;
  }

  getPoints() {
    return this.zeroOffset(this.buffer);
  }

  getLastChunk() {
    return this.zeroOffset(this.lastChunk);
  }

  zeroOffset(points) {
    return points.map(p => ({x: p.x - this.minX, y: p.y - this.minY, eos: p.eos}));
  }

  reset() {
    this.lastChunk = [];
    this.buffer = [];
    this.minX = 0;
    this.minY = 0;
    this.maxX = 0;
    this.maxY = 0;
  }
}


class HandwritingScreen extends React.Component {
  constructor(props) {
    super(props);

    this.resolution = 0.5;

    let defaultLogicalWidth = window.innerWidth * this.resolution;
    let defaultLogicalHeight = window.innerHeight * this.resolution;

    this.state = {
      scale: 1,
      points: [],
      text: "",
      done: true,
      bias: 0.5,
      showBackgroundPicker: false,
      showStrokeColorPicker: false,
      background: '#fff',
      strokeColor: '#000',
      primingText: "",
      primingSequence: []
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

    this.onProgressListener = null;
    this.onCompleteListener = null;
  }

  componentDidMount() {
    window.addEventListener('resize', this.adjustCanvasSize);
 
    this.onCompleteListener = (points) => {
      this.handleCompletion();
    };

    executor.subscribe("resultsReady", this.onCompleteListener);
  }

  handleCompletion() {
    this.setState({points: [], done: true, progress: 0});
  }

  componentDidUpdate() {
    console.log('Did update!!!');
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.adjustCanvasSize);
    if (this.workerListener) {
      worker.removeEventListener('message', this.workerListener);
    }

    executor.unsubscribe("resultsReady", this.onCompleteListener);
  }

  adjustCanvasSize() {
    this.forceUpdate();
  }

  handleClick() {
    if (this.state.bias < 0) {
      window.alert("Negative bias is not allowed!");
      return;
    }

    if (this.state.text.length >= 100) {
      window.alert("Text must contain fewer thatn 100 characters. Please, try again.");
      return;
    }
    
    this.setState({points: [], done: false});

    executor.runNewJob(this.state.text, this.state.bias, this.state.primingSequence, this.state.primingText);
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
    canvasConfig.update(color.hex, this.state.strokeColor);
  }

  handleChangeStrokeColor(color) {
    this.setState({strokeColor: color.hex});
    canvasConfig.update(this.state.background, color.hex);
  }
  render() {

    return (
      <div className="App">
        <Container>
          
          <textarea className="mb-2" placeholder="Enter text to generate a handwriting for" 
                    value={this.state.text} onChange={this.handleChange} maxLength="99">

          </textarea>
          <SettingsPanel bias={this.state.bias} primingText={this.state.primingText}
                         onChangeBackground={this.handleChangeBackground} 
                         onChangeStrokeColor={this.handleChangeStrokeColor}
                         onBiasChange={this.handleBiasChange}
                         onPrimingTextChange={e => this.setState({primingText: e.target.value})}
                         onPrimingSequenceChange={points => this.setState({primingSequence: points})}
                          />
          <Row className="mb-2">
            <Col>
              <Button onClick={this.handleClick} disabled={this.state.text.trim() === "" || !this.state.done}>
                Generate handwriting
              </Button>
            </Col>
          </Row>

          {!this.state.done && <InProgressPanel />}
          {this.state.done &&
            <ZoomButtonsGroup onZoomIn={this.handleZoomIn} onZoomOut={this.handleZoomOut} 
                              canZoomIn={this.canZoomIn()} canZoomOut={this.canZoomOut()} />
          }
        </Container>
        <div style={{ overflow: 'auto'}}>
          <MyCanvas scale={this.state.scale} />
        </div>
      </div>
    );
  }
}


class MyCanvas extends React.Component {
  constructor(props) {
    super(props);

    this.INITIAL_WIDTH = window.innerWidth / 2;
    this.INITIAL_HEIGHT = window.innerHeight / 8;
    this.RESOLUTION = 0.25;

    this.canvasRef = React.createRef();
    this.writer = null;
    this.virtualCanvas = null;

    this.onStartListener = null;
    this.onProgressListener = null;
    this.onConfigChangeListener = null;
  }

  componentDidMount() {
    console.log('canvas mounted')
    //window.addEventListener('resize', this.adjustCanvasSize);
    let canvas = this.canvasRef.current;
    this.writer = new CanvasDrawer(canvas, 5, '#fff', '#000');
    this.virtualCanvas = this.makeVirtualCanvas();

    this.onStartListener = () => {
      console.log("on Start!")
      canvas.width = this.INITIAL_WIDTH;
      canvas.height = this.INITIAL_HEIGHT;
      console.log(canvas.height);
      
      this.updateVisibleSize();

      this.virtualCanvas.reset();
      this.writer.reset();
    };

    this.onProgressListener = (points, progress) => {
      this.virtualCanvas.addChunk(points);
    };

    executor.subscribe("start", this.onStartListener);
    executor.subscribe("progressChanged", this.onProgressListener);

    this.onConfigChangeListener = (config) => {
      this.writer.setBackgroundColor(config.backgroundColor);
      this.writer.setLineColor(config.strokeColor);
      this.redrawCanvas();
    }

    canvasConfig.setListener(this.onConfigChangeListener);
  }

  componentWillUnMount() {
    console.log('canvas will be mounted')
    executor.unsubscribe("start", this.onStartListener);
    executor.unsubscribe("progressChanged", this.onProgressListener);
  }

  componentDidUpdate() {
    console.log('Did update mycanvas !!!');
  }

  makeVirtualCanvas() {
    const onArrival = virtualCanvas => {
      this.writer.draw(virtualCanvas.getLastChunk());
      this.writer.finish();
    }

    const onGeometryChange = virtualCanvas => {
      let canvas = this.canvasRef.current;

      let context = canvas.getContext('2d');

      if (canvas.width < virtualCanvas.getWidth() || canvas.height < virtualCanvas.getHeight()) {
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (canvas.width < virtualCanvas.getWidth()) {
          canvas.width = virtualCanvas.getWidth() * 1.5;
        }

        if (canvas.height < virtualCanvas.getHeight()) {
          canvas.height = virtualCanvas.getHeight() * 1.25;
        }

        this.updateVisibleSize();
        this.redrawCanvas();
      }
    }

    // todo: floating/dynamic scaling factor ()
    return new VirtualCanvas(onArrival, onGeometryChange, this.RESOLUTION);
  }

  updateVisibleSize() {
    let canvas = this.canvasRef.current;
    let scaledWidth = window.innerWidth * this.props.scale;
    let aspectRatio = canvas.width / canvas.height;
    let canvasHeight = Math.round(scaledWidth / aspectRatio);
    let canvasWidth = Math.round(scaledWidth);

    canvas.setAttribute("style", `width:${canvasWidth}px;height:${canvasHeight}px`);
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;
  }

  redrawCanvas() {
    console.log("redrawing")
    const minWidth = 2;

    let canvas = this.canvasRef.current;
    let scaledWidth = window.innerWidth * this.props.scale;
    let lineWidth = Math.floor(canvas.width / scaledWidth) + minWidth;

    let points = this.virtualCanvas.getPoints();

    this.writer.setWidth(lineWidth);
    this.writer.reset();
    this.writer.draw(points);
    this.writer.finish();
  }

  getAspectRatio() {
    let canvas = this.canvasRef.current;
    if (canvas) {
      return canvas.width / canvas.height;
    }
    return this.INITIAL_WIDTH / this.INITIAL_HEIGHT;
  }
  render() {
    let scaledWidth = window.innerWidth * this.props.scale;
    let canvasHeight = Math.round(scaledWidth / this.getAspectRatio());
    let canvasWidth = Math.round(scaledWidth);

    return (
      <canvas ref={this.canvasRef} width={this.INITIAL_WIDTH} height={this.INITIAL_HEIGHT} 
              style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`}} >

      </canvas>
    );
  }
}


class SettingsPanel extends React.Component {
  constructor(props) {
    super(props);

    this.itemRef = React.createRef();
    this.bodyRef = React.createRef();

    this.state = {
      width: window.innerWidth
    };
  }

  componentDidMount() {
    this.updateWidth();
  }

  componentDidUpdate() {
    this.updateWidth();
  }

  updateWidth() {
    const element = this.itemRef.current;
    const body = this.bodyRef.current;
    const computedStyle = window.getComputedStyle(body);
    const width = element.offsetWidth - (parseFloat(computedStyle.paddingLeft) + parseFloat(computedStyle.paddingRight));

    if (width !== this.state.width) {
      this.setState({width});
    }
  }

  render() {
    return (
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
                                  value={this.props.bias} min={0} max={100} step={0.1} 
                                  onChange={e => this.props.onBiasChange(e)} />
                  </Col>
              </Row>
              <MyColorPicker label='Background color' onChangeComplete={e => this.props.onChangeBackground(e) } />
              <MyColorPicker label='Stoke color' onChangeComplete={e => this.props.onChangeStrokeColor(e) } />
            </Form>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="1">
          <Accordion.Header ref={this.itemRef}>Custom style</Accordion.Header>
          <Accordion.Body ref={this.bodyRef}>
            <p>Priming is a easy and fast way to make a synthesis network adapt to your style of writing.
              You only need to provide an arbitrary piece of text and a corresponding handwriting. You can enter a 
              text into a text field below. Then, you need to create a handwritten version of the text by writing on a canvas.
            </p>
            <Form.Control type="text" value={this.props.primingText} placeholder="Enter a text used for priming"
                          onChange={e => this.props.onPrimingTextChange(e)} />
            <ScalableCanvas parentWidth={this.state.width} onPrimingSequenceChange={e => this.props.onPrimingSequenceChange(e)}/>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    );
  }
}


class MyColorPicker extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      show: false,
      color: '#ffffff'
    };

    this.handleChange = this.handleChange.bind(this);
    this.toggleShow = this.toggleShow.bind(this);
  }
  handleChange(color) {
    this.setState({color: color.hex});
    this.props.onChangeComplete(color);
  }

  toggleShow(e) {
    this.setState((state, cb) => ({
      show: !state.show
    }));
  }
  render() {
    let label = this.props.label || 'Color';
    let buttonText;
    if (this.state.show) {
      buttonText = 'Close picker';
    } else {
      buttonText = 'Choose a color';
    }

    return (
      <Row className="mb-2">
        <Col>
          <Form.Label className="mr-2">{label}:</Form.Label>
          <FilledSquare color={this.state.color} />
          <Button onClick={this.toggleShow}>
            {buttonText}
          </Button>
        </Col>
        {this.state.show &&
          <Col>
            <div style={{width: 'auto'}}>
              <SketchPicker
                style={{margin:'auto'}}
                color={ this.state.color }
                onChangeComplete={ this.handleChange }
              />
            </div>
          </Col>
        }
      </Row>
    );
  }
}


function FilledSquare(props) {
  let size = props.size || 25;
  return (
    <div style={{width: `${size}px`, height: `${size}px`, background: props.color, display: 'inline-block'}}>
    </div>
  );
}


class InProgressPanel extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      progress: 0
    };

    this.progressChangedListener = (results, progress) => {
      this.setState({progress});
    };
  }

  componentDidMount() {
    executor.subscribe("progressChanged", this.progressChangedListener);
  }

  componentWillUnmount() {
    executor.unsubscribe("progressChanged", this.progressChangedListener);
  }

  render() {

    const text = this.props.text || "Generating a handwriting, please wait...";
    return (
      <Row className="mb-2">
        <Col>
          <Row>
            <Col>
              <h4>{text}</h4>
            </Col>
          </Row>
          <Row>
            <Col>
              <ProgressBar now={this.state.progress} />
            </Col>
          </Row>
        </Col>
      </Row>
    );
  }
}


function ZoomButtonsGroup(props) {
  return (
    <Row className="mb-2">
      <Col>
        <ButtonGroup size="sm">
          <Button variant="secondary" onClick={e => props.onZoomIn(e)} disabled={!props.canZoomIn}>
            Zoom In
          </Button>
          <Button variant="secondary" onClick={e => props.onZoomOut(e)} disabled={!props.canZoomOut}>
            Zoom out
          </Button>
        </ButtonGroup>
      </Col>
    </Row>
  );
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
