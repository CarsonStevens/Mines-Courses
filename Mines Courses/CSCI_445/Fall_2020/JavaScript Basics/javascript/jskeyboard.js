/*jshint esversion: 8 */
$(document).ready(function() {

  const Player = (function(fps) {

    this.init(fps);

    $("#tutorials-container").append(this.canvas);
    this.LEFT = false;
    this.RIGHT = false;
    this.UP = false;
    this.DOWN = false;

    this.speed = 4;
    this.width = 50;
    this.height = 55;
    this.img = new Image();
    this.img.src = "../images/mario.png";

    window.addEventListener('keydown',this.checkKey.bind(this), false);
    window.addEventListener('keyup',this.removeKey.bind(this), false);
    this.render();
  });

  Player.prototype.checkKey = (function(e) {
      switch (e.keyCode) {
          case 37: this.LEFT = true; break;
          case 39: this.RIGHT = true; break;
          case 38: this.UP = true; break;
          case 40: this.DOWN = true; break;
          default: break;
      }
  });
  Player.prototype.removeKey = (function(e) {
      switch (e.keyCode) {
          case 37: this.LEFT = false; break;
          case 39: this.RIGHT = false; break;
          case 38: this.UP = false; break;
          case 40: this.DOWN = false; break;
          default: break;
      }
  });

  Player.prototype.render = (function() {
    if (this.nextFrame()) {
      this.clear();
      this.move();
      this.draw();
    }
    this.animation = window.requestAnimationFrame(this.render.bind(this));
  });

  Player.prototype.draw = (function() {

    this.ctx.drawImage(this.img, this.x, this.y, this.width, this.height);
  });

  Player.prototype.clear = (function() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  });

  Player.prototype.move = (function() {
    if(this.RIGHT) {
      if (this.x + this.width + this.speed <= this.canvas.width) this.x += this.speed;
      if (this.x + this.width + this.speed > this.canvas.width) this.x = this.canvas.width - this.width;
    }
    if(this.LEFT) {
      if (this.x - this.speed >= 0) this.x -= this.speed;
      if (this.x - this.speed < 0) this.x = 0;

    }
    if(this.UP) {
      if (this.y - this.speed >= 0) this.y -= this.speed;
      if (this.y - this.speed < 0) this.y = 0;
    }
    if(this.DOWN) {
      if (this.y + this.height + this.speed <= this.canvas.height) this.y += this.speed;
      if (this.y + this.height + this.speed > this.canvas.height) this.y = this.canvas.height -this.height;
    }
  });

  Player.prototype.init = (function(fps) {
    this.canvas = document.createElement("canvas");
    this.canvas.id = "jskeyboard";
    this.ctx = this.canvas.getContext("2d");
    this.canvas.width = 500;
    this.canvas.height = 300;

    this.x = this.canvas.width/2;
    this.y = this.canvas.height/2;

    window.requestAnimationFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;

    window.cancelAnimationFrame = window.cancelAnimationFrame || window.mozCancelAnimationFrame || window.webkitCancelAnimationFrame || this.window.msCancelAnimationFrame;

    this.fps = fps || 60;
    this.fpsInterval = 1000 / this.fps;
    this.now = Date.now();
    this.then = Date.now();
    this.startTime = this.then;

    this.render();
  });

  Player.prototype.nextFrame = function() {
    this.now = Date.now();
    this.elapsed = this.now - this.then;
    if (this.elapsed > this.fpsInterval) {
      this.then = this.now - (this.elapsed % this.fpsInterval);
      return true;
    } else {
      return false;
    }
  };

  const player = new Player(60);
});
