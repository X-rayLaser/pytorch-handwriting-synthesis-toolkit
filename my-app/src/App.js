import './App.css';
import React from 'react';
import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import { InputGroup } from 'react-bootstrap';
import Accordion from 'react-bootstrap/Accordion';
import CanvasDrawer from './drawing';
import ScalableCanvas from './ScalableCanvas';


class CanvasConfiguration {
  constructor(backgroundColor='#ffffff', strokeColor='#000000') {
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
        this.notify(e.data.event, e.data.results);
      } else if (e.data.event === "progressChanged") {
        this.notify(e.data.event, e.data.results, e.data.value);
      } else if (e.data.event === "aborted") {
        this.notify(e.data.event);
      }
    };

    worker.addEventListener('message', workerListener);
    worker.onerror = e => {
      this.notify("error", e);
    }
  }

  runNewJob(text, bias, primingSequence, primingText) {
    this.notify("start");
    worker.postMessage({
      event: "start",
      params: [text, bias, primingSequence, primingText]
    });
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
    worker.postMessage({
      event: "abort",
      params: []
    });
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

let worker = new Worker(new URL("./worker_manager.js", import.meta.url));
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
  constructor(onArrival, onGeometryChange, onGeometryError, scaleFactor=1) {
    this.buffer = [];
    this.lastChunk = [];
    this.scale = scaleFactor;
    this.minX = 0;
    this.minY = 0;
    this.maxX = 0;
    this.maxY = 0;

    this.onArrival = onArrival;
    this.onGeometryChange = onGeometryChange;
    this.onGeometryError = onGeometryError;

    this.MAX_AREA = 10000 * 1000;
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
      if (this.getWidth() * this.getHeight() > this.MAX_AREA) {
        this.onGeometryError(`Canvas area exceeded maximum value ${this.MAX_AREA}`);
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
    return this.buffer;
  }

  getLastChunk() {
    return this.lastChunk;
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

  reScale(scale) {
    //undo previous scaling and apply new scale to every point

    const update = z => z / this.scale * scale;

    const mapPoint = p => ({
      x: update(p.x),
      y: update(p.y),
      eos: p.eos
    });

    this.buffer = this.buffer.map(mapPoint);
    this.minX = update(this.minX);
    this.minY = update(this.minY);
    this.maxX = update(this.maxX);
    this.maxY = update(this.maxY);
    this.scale = scale;
  }
}


class HandwritingScreen extends React.Component {
  constructor(props) {
    super(props);

    this.resolution = 0.5;

    this.state = {
      scale: 1,
      points: [],
      text: "",
      error: "",
      done: true,
      showZoomButtons: false,
      operationStatus: "ready",
      bias: 0.5,
      background: '#ffffff',
      strokeColor: '#000000',
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
    this.handleCanvasError = this.handleCanvasError.bind(this);
    this.adjustCanvasSize = this.adjustCanvasSize.bind(this);

    this.onProgressListener = null;
    this.onCompleteListener = null;
    this.onAbortedListener = null;
    this.onErrorListener = null;
  }

  componentDidMount() {
    window.addEventListener('resize', this.adjustCanvasSize);
    window.addEventListener('changeorientation', this.adjustCanvasSize);

    this.onCompleteListener = (points) => {
      this.handleCompletion();
    };

    this.onAbortedListener = () => {
      this.setState({points: [], done: true, progress: 0, operationStatus: "ready"});
    }

    this.onErrorListener = errorEvent => {
      let message = "Something went wrong. Please, try again.";
      console.error(errorEvent);
      this.setState({done: true, progress: 0, operationStatus: "ready", error: message});
    };

    executor.subscribe("resultsReady", this.onCompleteListener);
    executor.subscribe("aborted", this.onAbortedListener);
    executor.subscribe("error", this.onErrorListener);
  }

  handleCompletion() {
    this.setState({showZoomButtons: true, done: true, progress: 0});
  }

  componentDidUpdate() {
    console.log('Did update!!!');
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.adjustCanvasSize);
    window.removeEventListener('changeorientation', this.adjustCanvasSize);

    if (this.workerListener) {
      worker.removeEventListener('message', this.workerListener);
    }

    executor.unsubscribe("resultsReady", this.onCompleteListener);
    executor.unsubscribe("aborted", this.onAbortedListener);
    executor.unsubscribe("error", this.onErrorListener);
  }

  adjustCanvasSize() {
    this.forceUpdate();
  }

  handleClick() {
    if (this.state.done) {
      this.setState({error: "", showZoomButtons: false});
      this.runNewJob();
      return;
    }

    if (this.state.operationStatus === "running") {
      this.setState({operationStatus: "halting"});
      executor.abort();
    }
  }

  runNewJob() {
    if (this.state.bias < 0) {
      window.alert("Negative bias is not allowed!");
      return;
    }

    if (this.state.text.length > 150) {
      window.alert("Text must not be longer than 150 characters. Please, try again.");
      return;
    }
    
    this.setState({points: [], done: false, operationStatus: "running"});

    executor.runNewJob(this.state.text, this.state.bias, this.state.primingSequence, this.state.primingText);
  }

  handleZoomIn() {
    if (this.canZoomIn()) {
      this.setState((state, cb) => ({scale: state.scale * 1.5}));
    }
  }

  handleZoomOut() {
    if (this.canZoomOut()) {
      this.setState((state, cb) => ({scale: state.scale / 1.5}));
    }
  }

  canZoomIn() {
    return this.state.scale < 3;
  }

  canZoomOut() {
    return this.state.scale > 1 / 3.;
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

  handleCanvasError(message) {
    this.setState({operationStatus: "halting", error: message});
    executor.abort();
  }

  render() {
    let disableButton;
    let buttonText;

    if (this.state.done) {
      buttonText = "Generate handwriting";
      disableButton = (this.state.text.trim() === "");
    } else {
      buttonText = "Abort";
      let operationStatus = this.state.operationStatus;

      if (operationStatus === "starting") {
        buttonText = "Generate handwriting";
        disableButton = true;
      } else if (operationStatus === "running") {
        buttonText = "Abort";
        disableButton = false;
      } else if (operationStatus === "halting") {
        buttonText = "Abort";
        disableButton = true;
      } else {
        console.error("Unknown status");
      }
    }

    return (
      <div className="App">
        <Container>
          <Form className="mb-2" onSubmit={e => e.preventDefault()}>
            <InputGroup className="mb-3">
              <InputGroup.Text id="text-addon">Text</InputGroup.Text>
              <Form.Control type="text" placeholder="Enter a text to generate a handwriting for" 
                            value={this.state.text} onChange={this.handleChange} maxLength="150"
                            aria-label="Text used for handwriting synthesis" aria-describedby="text-addon" />
            </InputGroup>
          </Form>
          <SettingsPanel bias={this.state.bias} primingText={this.state.primingText}
                         onChangeBackground={this.handleChangeBackground} 
                         onChangeStrokeColor={this.handleChangeStrokeColor}
                         backgroundColor={this.state.background}
                         strokeColor={this.state.strokeColor}
                         onBiasChange={this.handleBiasChange}
                         onPrimingTextChange={e => this.setState({primingText: e.target.value})}
                         onPrimingSequenceChange={points => this.setState({primingSequence: points})}
                          />
          <Row className="mb-2 text-center">
            <Col>
              <Button onClick={this.handleClick} disabled={disableButton}>
                {buttonText}
              </Button>
            </Col>
          </Row>

          {this.state.error && 
            <Row className="mb-2">
              <Col><Alert variant="danger">{this.state.error}</Alert></Col>
            </Row>
          }

          {!this.state.done && <InProgressPanel />}
          {this.state.done && this.state.showZoomButtons &&
            <ZoomButtonsGroup onZoomIn={this.handleZoomIn} onZoomOut={this.handleZoomOut} 
                              canZoomIn={this.canZoomIn()} canZoomOut={this.canZoomOut()} />
          }
        </Container>
        <MyCanvas scale={this.state.scale} onError={this.handleCanvasError} />
      </div>
    );
  }
}


class MyCanvas extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      scrollLeft: 0,
      scrollTop: 0,
      view_port_width: document.body.clientWidth
    };

    this.VEIW_PORT_HEIGHT = 400;

    this.CANVAS_WIDTH = 10000;
    this.CANVAS_HEIGHT = 1500;
    this.LINE_WIDTH = 4;

    this.paddingLeft = 0;
    this.paddingTop = 0;

    this.containerRef = React.createRef();
    this.canvasRef = React.createRef();

    this.writer = null;
    this.virtualCanvas = null;

    this.onStartListener = null;
    this.onProgressListener = null;
    this.onConfigChangeListener = null;
    this.onScrollListener = null;

    this.onResizeListener = null;
  }

  componentDidMount() {
    let containerDiv = this.containerRef.current;

    this.onScrollListener = e => {
      this.setState({
        scrollLeft: containerDiv.scrollLeft,
        scrollTop: containerDiv.scrollTop
      });
    };

    containerDiv.addEventListener('scroll', this.onScrollListener);

    this.onResizeListener = e => {
      let view_port_width = document.body.clientWidth;
      this.setState({view_port_width});
    };

    window.addEventListener("resize", this.onResizeListener);
    window.addEventListener("orientationchange", this.onResizeListener);

    let canvas = this.canvasRef.current;
    this.writer = new CanvasDrawer(canvas, 5, '#fff', '#000');
    this.virtualCanvas = this.makeVirtualCanvas();

    this.onStartListener = () => {
      canvas.width = this.state.view_port_width;
      canvas.height = this.VEIW_PORT_HEIGHT;
      this.virtualCanvas.reset();
      this.writer.reset();

      containerDiv.scrollLeft = 0;
      containerDiv.scrollTop = 0;
      this.paddingLeft = 0;
      this.paddingTop = 0;
      this.setState({scrollLeft: 0, scrollTop: 0});
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
    this.onResizeListener({});
  }

  componentWillUnMount() {
    executor.unsubscribe("start", this.onStartListener);
    executor.unsubscribe("progressChanged", this.onProgressListener);

    this.containerRef.current.removeEventListener('scroll', this.onScrollListener);
    
    window.removeEventListener("resize", this.onResizeListener);
    window.removeEventListener("orientationchange", this.onResizeListener);
  }

  componentDidUpdate() {
    let effectiveScale = this.effectiveScale();
    if (effectiveScale !== this.virtualCanvas.scale) {
      this.virtualCanvas.reScale(effectiveScale);
    }
    this.redrawCanvas();
  }

  makeVirtualCanvas() {
    const onArrival = virtualCanvas => {
      let points = this.toViewPortPoints(virtualCanvas.getLastChunk());
      this.writer.draw(points);
      this.writer.finish();
    }

    const onGeometryChange = virtualCanvas => {
      if (virtualCanvas.minX + this.paddingLeft < 0 || virtualCanvas.minY + this.paddingTop < 0) {
        this.paddingLeft = Math.abs(virtualCanvas.minX) * 1.25;
        this.paddingTop = Math.abs(virtualCanvas.minY) * 1.25;
        this.redrawCanvas();
      }

      let AUTOSCROLL_PIXELS = Math.round(this.state.view_port_width / 2);
      let maxX = virtualCanvas.maxX + this.paddingLeft;

      let containerDiv = this.containerRef.current;

      if (maxX + this.paddingLeft > this.state.scrollLeft + this.state.view_port_width) {
        if (maxX < this.state.view_port_width) {
          containerDiv.scrollLeft = 0;
        } else {
          containerDiv.scrollLeft = maxX - this.state.view_port_width + AUTOSCROLL_PIXELS;
        }

        this.setState({
          scrollLeft: containerDiv.scrollLeft
        });
      }
    }

    const onGeometryError = message => {
      this.props.onError(message);
    }

    
    let effectiveScale = this.effectiveScale();
    return new VirtualCanvas(onArrival, onGeometryChange, onGeometryError, effectiveScale);
  }

  redrawCanvas() {
    let points = this.toViewPortPoints(this.virtualCanvas.getPoints());
    this.writer.setPadding(this.paddingLeft, this.paddingTop);
    this.writer.setWidth(this.LINE_WIDTH);
    this.writer.reset();
    this.writer.draw(points);
    this.writer.finish();
  }
  
  effectiveScale() {
    // default scale 1 is too large
    return this.props.scale / 2;
  }
  toViewPortPoints(points) {
    let scrollLeft = this.state.scrollLeft;
    let scrollTop = this.state.scrollTop;
    return points.map(p => ({x: p.x - scrollLeft + this.paddingLeft, y: p.y - scrollTop + this.paddingTop, eos: p.eos}));
  }

  render() {
    return (
      <OverlayContainer width={this.state.view_port_width} height={this.VEIW_PORT_HEIGHT}>
        <div ref={this.containerRef} style={{position: 'absolute', width: this.state.view_port_width, height: this.VEIW_PORT_HEIGHT, top:0, left:0, overflow: 'auto'}}>
          <div style={{width: this.CANVAS_WIDTH, height: this.CANVAS_HEIGHT}}></div>
        </div>
        <canvas ref={this.canvasRef} width={this.state.view_port_width} height={this.VEIW_PORT_HEIGHT} 
                style={{ position: 'absolute', width: `${this.state.view_port_width}px`, height: `${this.VEIW_PORT_HEIGHT}px`, left: 0, top: 0, zIndex: -1}} >

        </canvas>
      </OverlayContainer>
    );
  }
}

function OverlayContainer(props) {
  return (
    <div style={{position:'relative', width: props.width, height: props.height}}>
      {props.children}
    </div>
  );
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
            <Form onSubmit={e => e.preventDefault()}>
              <Form.Group className="mb-3" controlId="formBias">
                <InputGroup className="mb-3">
                  <InputGroup.Text id="bias-addon">Bias</InputGroup.Text>
                    <Form.Control type="number" value={this.props.bias} min={0} max={100} step={0.1} 
                      onChange={e => this.props.onBiasChange(e)}
                      placeholder="Bias"
                      aria-label="Bias"
                      aria-describedby="bias-addon"
                    />
                </InputGroup>
                <Form.Text className="text-muted">
                  Higher values result in a cleaner, nicer looking handwriting, while lower values result in less readable but more diverse samples
                </Form.Text>
              </Form.Group>
              <MyColorPicker label='Background color' color={this.props.backgroundColor} onChangeComplete={e => this.props.onChangeBackground(e) } />
              <MyColorPicker label='Handwriting color' color={this.props.strokeColor} onChangeComplete={e => this.props.onChangeStrokeColor(e) } />
            </Form>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="1">
          <Accordion.Header ref={this.itemRef}>Custom style</Accordion.Header>
          <Accordion.Body ref={this.bodyRef}>
            <p>
              Priming is an easy and fast way to make a synthesis network adapt to your style of writing. 
              You only need to provide a short piece of text and corresponding handwriting. 
              You can enter a text into a text field below. 
              Then, you need to create a handwritten version of the text by writing on a canvas. 
              As soon as you do the steps, the neural network will be able to mimic (to some extent) your style of writing.
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
    this.handleChange = this.handleChange.bind(this);
  }
  handleChange(color) {
    this.props.onChangeComplete(color);
  }

  render() {
    let label = this.props.label || 'Color';

    return (
      <Form.Group as={Row}>
        <Col xs={2}>
          <Form.Label htmlFor="exampleColorInput">{label}</Form.Label>
        </Col>
      <Col xs={1}>
      <Form.Control
        type="color"
        id="exampleColorInput"
        defaultValue={this.props.color}
        title="Choose a color"
        onChange={e => this.handleChange({hex: e.target.value})}
      />
      </Col>
      </Form.Group>
    );
  }
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
    <Row className="mb-2 text-center">
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
