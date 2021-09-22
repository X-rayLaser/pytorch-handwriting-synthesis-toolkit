export default class CanvasDrawer {
    constructor(canvas, lineWidth, background, strokeColor) {
      this.wasEos = false;
      this.canvas = canvas;
      this.background = background;
      this.strokeColor = strokeColor;

      this.context = canvas.getContext('2d');
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
      const ctx = this.context;
  
      points.forEach(p => {
        if (this.wasEos) {
          ctx.moveTo(p.x, p.y);
          this.wasEos = false;
        } else {
          ctx.lineTo(p.x, p.y);
        }
  
        if (p.eos == 1) {
          this.wasEos = true;
        }
      });
    }
  
    finish() {
      this.context.stroke();
    }

    reset() {
      this.context.fillStyle = this.background;
      this.context.strokeStyle = this.strokeColor;

      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (this.background !== '#fff') {
        this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
      }
      this.context.beginPath();
    }

    setWidth(lineWidth) {
      this.context.lineWidth = lineWidth;
    }

    setBackgroundColor(color) {
      this.background = color;
    }

    setLineColor(color) {
      this.strokeColor = color;
    }
}
