(this["webpackJsonpmy-app"]=this["webpackJsonpmy-app"]||[]).push([[0],{16:function(t,e,n){var a=n(26),i=["startWorker"];t.exports=function(){var t=new Worker(n.p+"d47adc4009754b8f31be.worker.js",{name:"[hash].worker.js"});return a(t,i),t}},24:function(t,e,n){},25:function(t,e,n){},29:function(t,e,n){"use strict";n.r(e);var a=n(1),i=n.n(a),s=n(15),h=n.n(s),o=(n(24),n(3)),r=n(18),c=n(17),d=n(13),l=n(10),u=n(11),g=(n(25),n(12)),v=n(19),w=n(16),m=n.n(w),j=n(0),b=function(){function t(e,n,a,i){Object(l.a)(this,t),this.wasEos=!1,this.context=e.getContext("2d",{alpha:!1}),this.marginX=n,this.marginY=a,this.context.clearRect(0,0,e.width,e.height),this.context.lineWidth=i||30,this.context.beginPath()}return Object(u.a)(t,[{key:"draw",value:function(t){var e=this,n=this.marginX,a=this.marginY,i=this.context;t.forEach((function(t){e.wasEos?(i.moveTo(t.x-n,t.y-a),e.wasEos=!1):i.lineTo(t.x-n,t.y-a),1==t.eos&&(e.wasEos=!0)}))}},{key:"finish",value:function(){this.context.stroke()}}]),t}();function y(t){var e=1e5,n=1e5,a=-1e4,i=-1e4;return t.forEach((function(t){e=Math.min(e,t.x),n=Math.min(n,t.y),a=Math.max(a,t.x),i=Math.max(i,t.y)})),{minX:e,minY:n,width:a-e,height:i-n}}var x=function(t){Object(r.a)(n,t);var e=Object(c.a)(n);function n(t){var a;Object(l.a)(this,n),(a=e.call(this,t)).canvasRef=i.a.createRef();var s=window.innerWidth;return a.state={points:[],text:"",done:!0,bias:.5,canvasHeight:window.innerWidth/(s/1e3),canvasWidth:s,geometry:{x:200,y:200,width:s,height:1e3}},a.context=null,a.handleClick=a.handleClick.bind(Object(o.a)(a)),a.handleZoomIn=a.handleZoomIn.bind(Object(o.a)(a)),a.handleZoomOut=a.handleZoomOut.bind(Object(o.a)(a)),a.handleCancel=a.handleCancel.bind(Object(o.a)(a)),a.handleChange=a.handleChange.bind(Object(o.a)(a)),a.handleBiasChange=a.handleBiasChange.bind(Object(o.a)(a)),a.adjustCanvasSize=a.adjustCanvasSize.bind(Object(o.a)(a)),a.worker=null,a}return Object(u.a)(n,[{key:"resetGeometry",value:function(){var t=window.innerWidth;this.setState({canvasHeight:window.innerWidth/(t/1e3),canvasWidth:t,geometry:{x:200,y:200,width:t,height:1e3}})}},{key:"getAspectRatio",value:function(){return this.state.geometry.width/this.state.geometry.height}},{key:"componentDidMount",value:function(){var t=this;window.addEventListener("resize",this.adjustCanvasSize),this.context=this.canvasRef.current.getContext("2d"),this.worker=m()(),this.worker.addEventListener("message",(function(e){"resultsReady"===e.data.event&&t.setState({points:e.data.results,done:!0,progress:0}),"progressChanged"===e.data.event&&t.setState((function(t,n){var a=[].concat(Object(d.a)(t.points),Object(d.a)(e.data.results)),i=y(a);if(i.width=Math.max(window.innerWidth,t.geometry.width,i.width),i.height=Math.max(window.innerHeight,t.geometry.height,i.height),i.width=Math.min(i.width,1e4),i.height=Math.min(i.height,2e3),i.width>t.geometry.width){var s=Math.round(t.geometry.width/2);i.width=t.geometry.width+s}return{geometry:i,canvasWidth:window.innerWidth,canvasHeight:window.innerWidth/(i.width/i.height),progress:e.data.value,points:a}}))}))}},{key:"componentDidUpdate",value:function(){this.updateCanvas()}},{key:"componentWillUnmount",value:function(){window.removeEventListener("resize",this.adjustCanvasSize)}},{key:"adjustCanvasSize",value:function(){this.setState({canvasWidth:window.innerWidth,canvasHeight:window.innerWidth/this.getAspectRatio()})}},{key:"updateCanvas",value:function(){var t=this.canvasRef.current,e=this.state.geometry.minX,n=this.state.geometry.minY,a=Math.floor(this.state.geometry.width/this.state.canvasWidth)+5,i=new b(t,e,n,a);i.draw(this.state.points),i.finish()}},{key:"handleClick",value:function(){this.state.bias<0?window.alert("Negative bias is not allowed!"):this.state.text.length<6?window.alert("Text must contain at least 6 characters. Please, try again."):this.state.text.length>=50?window.alert("Text must contain fewer thatn 50 characters. Please, try again."):(this.resetGeometry(),this.setState({points:[],done:!1}),this.worker.startWorker(this.state.text,this.state.bias))}},{key:"handleZoomIn",value:function(){this.setState((function(t,e){return{canvasWidth:2*t.canvasWidth,canvasHeight:2*t.canvasHeight}}))}},{key:"handleZoomOut",value:function(){this.setState((function(t,e){return{canvasWidth:Math.round(t.canvasWidth/2),canvasHeight:Math.round(t.canvasHeight/2)}}))}},{key:"handleCancel",value:function(){this.worker&&(this.worker.terminate(),this.setState({points:[],done:!0,progress:0}))}},{key:"handleChange",value:function(t){this.setState({text:t.target.value})}},{key:"handleBiasChange",value:function(t){try{var e=parseFloat(t.target.value);e>=0&&this.setState({bias:e})}catch(t){console.error(t)}}},{key:"render",value:function(){return Object(j.jsxs)("div",{className:"App",children:[Object(j.jsx)("textarea",{placeholder:"Enter text for a handwriting",value:this.state.text,onChange:this.handleChange}),Object(j.jsxs)("details",{children:[Object(j.jsx)("summary",{children:"Settings"}),Object(j.jsx)("label",{children:"Bias"}),Object(j.jsx)("input",{type:"number",value:this.state.bias,min:0,max:100,step:.1,onChange:this.handleBiasChange})]}),Object(j.jsx)("div",{children:Object(j.jsx)(g.a,{onClick:this.handleClick,disabled:""===this.state.text.trim()||!this.state.done,children:"Generate"})}),!this.state.done&&Object(j.jsx)("div",{children:"Generating a handwriting, please wait..."}),!this.state.done&&Object(j.jsx)(v.a,{now:this.state.progress}),this.state.done&&this.state.points.length>0&&Object(j.jsxs)("div",{children:[Object(j.jsx)(g.a,{onClick:this.handleZoomIn,children:"Zoom In"}),Object(j.jsx)(g.a,{onClick:this.handleZoomOut,children:"Zoom out"})]}),Object(j.jsx)("div",{style:{overflow:"auto"},children:Object(j.jsx)("canvas",{ref:this.canvasRef,width:this.state.geometry.width,height:this.state.geometry.height,style:{width:"".concat(this.state.canvasWidth,"px"),height:"".concat(this.state.canvasHeight,"px")}})})]})}}]),n}(i.a.Component);var f=function(){return Object(j.jsxs)("div",{children:[Object(j.jsxs)("h4",{style:{textAlign:"center"},children:["This is a handwriting synthesis demo. It is a Javascript port of",Object(j.jsx)("a",{href:"https://github.com/X-rayLaser/pytorch-handwriting-synthesis-toolkit",children:" pytorch-handwriting-synthesis-toolkit"})," repository."]}),Object(j.jsx)(x,{})]})},p=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,30)).then((function(e){var n=e.getCLS,a=e.getFID,i=e.getFCP,s=e.getLCP,h=e.getTTFB;n(t),a(t),i(t),s(t),h(t)}))};n(28);h.a.render(Object(j.jsx)(i.a.StrictMode,{children:Object(j.jsx)(f,{})}),document.getElementById("root")),p()}},[[29,1,2]]]);
//# sourceMappingURL=main.1e26a65b.chunk.js.map