$(document).ready(function() {
  // Word Particle Effects
  /**
   * [WordParticle]: Create the interactive particle text. Parameters for
   *                 overall drawing are at the top of the document.
   * @param       {[double]} x [destination for x coord]
   * @param       {[double]} y [destination for y coord]
   */
  function WordParticle(x, y) {
    this.x = Math.random() * ww;
    this.y = Math.random() * wh;
    this.dest = {
      x: x,
      y: y
    };
    // size of word particles
    this.r = Math.random() * maxParticleRadius + minParticleRadius;
    this.vx = (Math.random() - 0.5) * 2000;
    this.vy = (Math.random() - 0.5) * 2000;
    this.accX = 0;
    this.accY = 0;
    this.friction = Math.random() * 0.005 + 0.93;

    this.color = colors[Math.floor(Math.random() * 6)];
  }

  /**
   * Makes each word particle move closer to its destination x and y
   * @return {[bool]} [Should return false if the movement path is invalid and
   *                   the particle should not be printed]
   */
  WordParticle.prototype.move = function() {
    this.accX = (this.dest.x - this.x) / 700;
    this.accY = (this.dest.y - this.y) / 700;
    this.vx += this.accX;
    this.vy += this.accY;
    this.vx *= this.friction;
    this.vy *= this.friction;

    this.x += this.vx;
    this.y += this.vy;

    let a = this.x - mouse.x;
    let b = this.y - mouse.y;

    let distance = Math.sqrt(a * a + b * b);
    if (distance < (radius * 70)) {
      this.accX = (this.x - mouse.x) / 5;
      this.accY = (this.y - mouse.y) / 5;
      this.vx += this.accX;
      this.vy += this.accY;
    }

    this.render();
    return true;
  };

  /**
   * Draws the word particles
   */
  WordParticle.prototype.render = function() {
    canvas.fillStyle = this.color;
    canvas.beginPath();
    canvas.arc(this.x, this.y, this.r, Math.PI * 2, false);
    canvas.fill();
  };

  function update() {
    now = Date.now();
    elapsed = now - then;
    if (elapsed > fpsInterval) {
      then = now - (elapsed % fpsInterval);
      canvas.clearRect(0, 0, ww, wh);
      word_particles = word_particles.filter(function(wp) {
        return wp.move();
      });
    }
    requestAnimationFrame(update.bind(this));
  }


  /**
   * [createWordCanvas]: Creates the word particle canvas
   * @param  {[Object]} properties [attributes for canvas size]
   * @return {[canvas]}            [new canvas object]
   */
  function createWordCanvas(properties) {
    let canvasContainer = $("#canvas_intro_container");
    let canvasElm = document.createElement('canvas');
    $(canvasContainer).append(canvasElm);

    if (typeof(canvasElm) !== 'undefined') {
      canvasElm.width = properties.width;
      canvasElm.height = properties.height;
      canvasElm.style.position = "absolute";
      // canvasElm.id = "layer"
      // canvasElm.style.zIndex = "4";
      let context = canvasElm.getContext('2d');
      return {
        canvas: canvasElm,
        context: context
      };
    }
  }

  // Frame rate controls
  let fps, fpsInterval, startTime, now, then, elapsed;
  fps = 60;

  const mouse = {
    x: 0,
    y: 0
  };
  let ww = $("#canvas_intro_container").width() || window.innerWidth;
  let wh = $("#canvas_intro_container").height() || window.innerHeight;
  let canvas, textDetails, word_particles, font, text, wordHeight, c, maxParticleRadius, minParticleRadius, textSize, letterSpacing, colors, mouseRadius, radius, currentWord, canvasElm, density;
  window.requestAnimationFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;


  const initScene = () => {
    ww = window.innerWidth;
    wh = $("#canvas_intro_container").height() || window.innerHeight;
    textDetails = JSON.parse(document.querySelector('*[data-wordDetails]').dataset.worddetails);
    mouseRadius = textDetails.mouseRadius || ww / 750;
    radius = mouseRadius;
    density = 1 / textDetails.density || 1;
    colors = textDetails.colors || ["#E4DB77", "#EF60DD", "#1656A9", "#C31445", "#AD7FB6"];
    text = textDetails.content || ["Add Content"];
    text.forEach((item, i) => {
      text[i] = item.replace(/_/g, "'");
    });
    font = textDetails.font || "Arial";
    WebFont.load({
      google: {
        families: [font]
      }
    });
    currentWord = 0;
    textSize = textDetails.textSize || ww / 10;
    letterSpacing = ww / 100 + "px";
    minParticleRadius = ww * 0.004;
    maxParticleRadius = textDetails.maxSize || ww / 300;
    textChangeInterval = textDetails.changeInterval || wh * 15.5;
    wordHeight = textDetails.textHeight || 15;


    setInterval(function() {
      changeWordParticles();
    }, textChangeInterval);

    function startAnimating(fps) {
      changeWordParticles();
      fpsInterval = 1000 / fps;
      then = Date.now();
      startTime = then;
      update();
    }
    startAnimating(fps);
  }

  /**
   * [changeWordParticles]: Gets the data for the coords for each word particle
   */
  function changeWordParticles() {
    word_particles = [];
    update();
    canvas.clearRect(0, 0, window.innerWidth, window.innerHeight);
    c.canvas.style.letterSpacing = letterSpacing;
    canvas.font = "bold " + (textSize) + "px " + font.toLowerCase().trim();
    let textWidth = canvas.measureText(text[currentWord]).width;
    if (textWidth > window.innerWidth - 50) {
      canvas.font = "bold " + parseInt(textSize * (window.innerWidth - 50) / textWidth) + font.toLowerCase().trim();
      textWidth = canvas.measureText(text[currentWord]).width;
    }
    canvas.textBaseline = "middle";
    canvas.textAlign = "center";
    canvas.fillText(text[currentWord], window.innerWidth / 2, window.innerHeight / wordHeight);

    let data = [];
    data = canvas.getImageData(0, 0, window.innerWidth, window.innerWidth).data;
    canvas.globalCompositeOperation = "screen";
    for (let i = 0; i < window.innerWidth; i += Math.round(window.innerWidth / 150 * density)) {
      for (let j = 0; j < window.innerHeight; j += Math.round(window.innerWidth / 150 * density)) {
        if (data[((i + j * window.innerWidth) * 4) + 3] > 150) {
          word_particles.push(new WordParticle(i, j));
        }
      }
    }
    canvas.clearRect(0, 0, window.innerWidth, window.innerHeight);

    currentWord++;
    currentWord = currentWord % text.length;

  }

  /**
   * [onMouseClick]: Change radius of mouse for disturbing word particles
   */
  function onMouseClick() {
    radius += 0.5;
    if (radius > 3) {
      radius = mouseRadius;
    }
  }

  /**
   * [onMouseMove]: get user mouse x and y coords when mouse moves
   * @param  {[event]} e [User input for client mouse X and Y]
   */
  function onMouseMove(e) {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  }

  /**
   * [onTouchMove]: gets the touch screen coords for X and Y
   * @param  {[event]} e [User inputer for client touch X and Y]
   */
  function onTouchMove(e) {
    if (e.touches.length > 0) {
      mouse.x = e.touches[0].clientX;
      mouse.y = e.touches[0].clientY;
    }
  }

  /**
   * [onTouchEnd]: Makes sure that after the user isn't touching that the Words
   *               aren't disturbed
   * @param  {[event]} e [User input]
   */
  function onTouchEnd(e) {
    mouse.x = -9999;
    mouse.y = -9999;
  }

  function resize(e) {
    $("#canvas_intro_container canvas").width($("#canvas_intro_container").width());
    canvas.height = $("#canvas_intro_container canvas").height($("#canvas_intro_container").height());
    // maxParticleRadius = ww / 300;
    changeWordParticles();
  }
  // Add listeners for interactive particle text
  window.addEventListener("resize", initScene);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("touchmove", onTouchMove);
  window.addEventListener("click", onMouseClick);
  window.addEventListener("touchend", onTouchEnd);
  c = createWordCanvas({
    width: ww,
    height: wh
  });
  canvas = c.context;
  initScene();
});
