import React from 'react';
import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';

export default class ScalableCanvas extends React.Component {
    constructor(props) {
        super(props);

        this.canvasRef = React.createRef();
        this.containerRef = React.createRef();

        this.state = {
            points: [],
            scale: 1,
            width: 5000,
            height: 750,
            recordingMode: false
        };

        this.cellSize = 200;

        this.handleZoomIn = this.handleZoomIn.bind(this);
        this.handleZoomOut = this.handleZoomOut.bind(this);
        this.handleClear = this.handleClear.bind(this);

        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
    }

    canZoomIn() {
        return this.state.scale < 10;
    }
    
    canZoomOut() {
        return this.state.scale > 0.1;
    }

    componentDidMount() {
        this.updateCanvas();
    }

    componentDidUpdate() {
        this.updateCanvas();
    }

    updateCanvas() {
        const canvas = this.canvasRef.current;
        this.renderGrid(canvas);
        this.renderHandwriting(canvas);
    }

    renderGrid(canvas) {
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.lineWidth = 10;
        context.strokeStyle = '#ccc';
        context.beginPath();

        const cellSize = this.cellSize;
        for (let y = 0; y < this.state.height; y += cellSize) {
            context.moveTo(0, y);
            context.lineTo(this.state.width, y);
        }

        for (let x = 0; x < this.state.width; x += cellSize) {
            context.moveTo(x, 0);
            context.lineTo(x, this.state.height);
        }

        context.stroke();
    }

    renderHandwriting(canvas) {
        const context = canvas.getContext('2d');
        context.lineWidth = 10;
        context.strokeStyle = '#000';
        context.beginPath();

        let newStroke = true;

        this.state.points.forEach(p => {
            if (newStroke) {
                context.moveTo(p.x, p.y);
            } else {
                context.lineTo(p.x, p.y);
            }
            
            newStroke = p.eos;
        });

        context.stroke();
    }

    getCoordinates(e) {
        const canvas = this.canvasRef.current;
        let stretchFactor = canvas.width / (this.props.parentWidth * this.state.scale);

        let effectiveX = e.nativeEvent.offsetX * stretchFactor;
        let effectiveY = e.nativeEvent.offsetY * stretchFactor;
        return {
            x: effectiveX,
            y: effectiveY
        };
    }

    handleMouseDown(e) {
        const dataPoint = this.getCoordinates(e);
        dataPoint.eos = false;
        this.setState((state, cb) => {
            return {
                recordingMode: true,
                points: [...state.points, dataPoint]
            };
        }, () => {
            this.props.onPrimingSequenceChange(this.state.points);
        });
    }

    handleMouseMove(e) {
        if (!this.state.recordingMode) {
            return;
        }

        const dataPoint = this.getCoordinates(e);
        dataPoint.eos = false;

        this.setState((state, cb) => ({
            points: [...state.points, dataPoint]
        }), () => {
            this.props.onPrimingSequenceChange(this.state.points);
        });
    }

    handleMouseUp(e) {
        const dataPoint = this.getCoordinates(e);
        dataPoint.eos = true;
        this.setState((state, cb) => {
            return {
                recordingMode: false,
                points: [...state.points, dataPoint]
            };
        }, () => {
            this.props.onPrimingSequenceChange(this.state.points);
        });
    }

    handleZoomIn() {
        this.setState((state, cb) => {
            if (state.scale < 10) {
                return {
                    scale: state.scale * 2
                }
            } else {
                return {
                    scale: state.scale
                };
            }
        });
    }
    handleZoomOut() {
        this.setState((state, cb) => {
            if (state.scale > 0.1) {
                return {
                    scale: state.scale / 2
                }
            } else {
                return {};
            }
        });
    }

    handleClear() {
        this.setState({points: []});
    }

    getAspectRatio() {
        return this.state.width / this.state.height;
    }

    render() {
        let scaledWidth = this.props.parentWidth * this.state.scale;
        let styleWidth = Math.round(scaledWidth);
        let styleHeight = Math.round(scaledWidth / this.getAspectRatio());

        return (
            <div ref={this.containerRef}>
                <ButtonGroup size="sm">
                    <Button variant="secondary" onClick={this.handleZoomIn} disabled={!this.canZoomIn()}>
                        Zoom In
                    </Button>
                    <Button variant="secondary" onClick={this.handleZoomOut} disabled={!this.canZoomOut()}>
                        Zoom out
                    </Button>
                    <Button variant="secondary" onClick={this.handleClear}>
                        Clear
                    </Button>
                </ButtonGroup>
                <div style={{overflow: 'auto'}}>
                    <canvas ref={this.canvasRef} width={this.state.width} height={this.state.height} 
                            onMouseDown={this.handleMouseDown} onMouseMove={this.handleMouseMove} onMouseUp={this.handleMouseUp}
                            style={{width: `${styleWidth}px`, height: `${styleHeight}px`}}>

                    </canvas>
                </div>
            </div>
        );
    }
}