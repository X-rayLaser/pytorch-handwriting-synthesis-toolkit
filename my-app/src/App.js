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

let worker = new Worker(new URL("./worker.js", import.meta.url));


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
          <SettingsPanel bias={this.state.bias} 
                         onChangeBackground={this.handleChangeBackground} 
                         onChangeStrokeColor={this.handleChangeStrokeColor}
                         onBiasChange={this.handleBiasChange} />
          <Row className="mb-2">
            <Col>
              <Button onClick={this.handleClick} disabled={this.state.text.trim() === "" || !this.state.done}>
                Generate handwriting
              </Button>
            </Col>
          </Row>

          {!this.state.done && <InProgressPanel progress={this.state.progress} />}
          {this.state.done && this.state.points.length > 0 &&
            <ZoomButtonsGroup onZoomIn={this.handleZoomIn} onZoomOut={this.handleZoomOut} 
                              canZoomIn={this.canZoomIn()} canZoomOut={this.canZoomOut()} />
          }
        </Container>
        <div style={{ overflow: 'auto'}}>
          <canvas ref={this.canvasRef} width={this.state.geometry.width} height={this.state.geometry.height} 
                  style={{ width: `${canvasWidth}px`, height: `${canvasHeight}px`}} >

          </canvas>
        </div>
      </div>
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
            <ScalableCanvas parentWidth={this.state.width}/>
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


function InProgressPanel(props) {
  const text = props.text || "Generating a handwriting, please wait...";
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
            <ProgressBar now={props.progress} />
          </Col>
        </Row>
      </Col>
    </Row>
  );
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
