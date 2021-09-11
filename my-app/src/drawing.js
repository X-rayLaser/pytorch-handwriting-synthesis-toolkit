export default class CanvasDrawer {
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
