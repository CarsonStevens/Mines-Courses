/*jshint esversion: 8 */
$(document).ready(function() {

  const Smiley = (function(fps) {
    this.state = "happy";
    this.faceColor = "rgb(255, 243, 88)";
    this.faceRadius = 100;
    this.eyeColor = "rgb(23, 169, 232)";
    this.eyeRadius = 10;

    this.sadPhrase = "Make me sad...";
    this.happyPhrase = "Make me HAPPY!";
    this.smileHeight = 30;
    this.smileWidth = 50;
    this.smileIntensity = 3;

    this.mouse = {
        x : 0, y : 0,  // coordinates
        lastX : 0, lastY : 0, // last frames mouse position
        b1 : false, b2 : false, b3 : false, // buttons
        buttonNames : ["b1","b2","b3"],  // named buttons
    };

    this.init(fps);
    $("#tutorials-container").append(this.canvas);
  });

  Smiley.prototype.init = (function(fps) {
    this.canvas = document.createElement("canvas");
    this.canvas.id = "jsmouse";
    this.ctx = this.canvas.getContext("2d");
    this.canvas.width = 500;
    this.canvas.height = 300;

    // this.canvas.style.width = "100%";
    // this.canvas.style.height = "100%";

    this.x = this.canvas.width/2;
    this.y = this.canvas.height/2;
    this.smileOffsetY = 30;
    this.smileY = this.y - this.smileHeight + this.smileOffsetY;
    this.smileX = this.x;
    this.smileBegin = Math.PI;
    this.smileEnd = Math.PI * 2;

    window.requestAnimationFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;

    window.cancelAnimationFrame = window.cancelAnimationFrame || window.mozCancelAnimationFrame || window.webkitCancelAnimationFrame || this.window.msCancelAnimationFrame;

    this.fps = fps || 60;
    this.fpsInterval = 1000 / this.fps;
    this.now = Date.now();
    this.then = Date.now();
    this.startTime = this.then;

    $(this.canvas).on('click', this.checkClick.bind(this));
    $("#smiley").on('click', this.changeState.bind(this));
    this.render();
  });

  Smiley.prototype.normalizeMouse = (function(event) {
      this.bounds = this.canvas.getBoundingClientRect();
      // get the mouse coordinates, subtract the canvas top left and any scrolling
      this.mouse.x = event.pageX - this.bounds.left;
      this.mouse.y = event.pageY - this.bounds.top;
      // first normalize the mouse coordinates from 0 to 1 (0,0) top left
     // off canvas and (1,1) bottom right by dividing by the bounds width and height
      this.mouse.x /=  this.bounds.width;
      this.mouse.y /=  this.bounds.height;
      // then scale to canvas coordinates by multiplying the normalized coords with the canvas resolution

     this.mouse.x *= this.canvas.width;
     this.mouse.y *= this.canvas.height;
  });


  Smiley.prototype.render = (function() {
    if (this.nextFrame()) {
      this.clear();
      this.renderFace();
      this.renderEyes();
      this.smile();
    }
    this.animation = window.requestAnimationFrame(this.render.bind(this));
  });

  Smiley.prototype.nextFrame = function() {
    this.now = Date.now();
    this.elapsed = this.now - this.then;
    if (this.elapsed > this.fpsInterval) {
      this.then = this.now - (this.elapsed % this.fpsInterval);
      return true;
    } else {
      return false;
    }
  };

  Smiley.prototype.clear = (function() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  });


  Smiley.prototype.renderFace = (function() {
    this.ctx.beginPath();
    this.ctx.arc(this.x, this.y, this.faceRadius, 0, 2 * Math.PI);
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = "rgb(0,0,0)";
    this.ctx.fillStyle = this.faceColor;
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.closePath();
  });

  Smiley.prototype.renderEyes = (function(x) {
    this.ctx.beginPath();
    this.ctx.arc(this.x-30, this.y-30, this.eyeRadius, 0, 2 * Math.PI);
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = "rgb(0,0,0)";
    this.ctx.fillStyle = this.eyeColor;
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.closePath();

    this.ctx.beginPath();
    this.ctx.arc(this.x+30, this.y-30, this.eyeRadius, 0, 2 * Math.PI);
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = "rgb(0,0,0)";
    this.ctx.fillStyle = this.eyeColor;
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.closePath();
  });

  Smiley.prototype.smile = (function() {
    this.ctx.beginPath();
    this.ctx.arc(this.smileX, this.smileY, this.smileWidth, this.smileBegin-Math.PI/this.smileIntensity, this.smileEnd+Math.PI/this.smileIntensity, true);
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = "rgb(0,0,0)";
    this.ctx.stroke();
    this.ctx.closePath();
  });

  Smiley.prototype.checkClick = (function(e) {
    this.normalizeMouse(e);
    let dx = this.x - this.mouse.x;
    let dy = this.y - this.mouse.y;
    let dist = Math.abs(Math.sqrt(dx*dx + dy*dy));
    if (dist <= this.faceRadius) {
      this.changeState();
    }
  });

  Smiley.prototype.changeState = (function() {
    this.smileIntensity += 0.5;
    if (this.state == "happy") {
      this.state = "sad";
      this.smileBegin = 0;
      this.smileEnd = Math.PI;
      this.smileY = this.y + this.smileHeight + this.smileOffsetY;
      $("#smiley").attr({"data-phrase" : this.happyPhrase});
    } else {
      this.state = "happy";
      this.smileBegin = Math.PI;
      this.smileEnd = Math.PI * 2;
      this.smileY = this.y - this.smileHeight + this.smileOffsetY;
      $("#smiley").attr({"data-phrase" : this.sadPhrase});
    }
  });

  const SmileyFace = new Smiley(60);
});
