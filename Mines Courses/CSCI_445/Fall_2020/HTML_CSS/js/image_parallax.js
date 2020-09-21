$(document).ready(function() {
  $.fn.parallax = function(resistance, mouse) {
    $el = $(this);
    gsap.to($el, 0.001, {
      x: ((mouse.clientX - window.innerWidth / 2) / resistance) * 3,
      y: -((mouse.clientY - window.innerHeight / 2) / resistance)
    });
  };
});
