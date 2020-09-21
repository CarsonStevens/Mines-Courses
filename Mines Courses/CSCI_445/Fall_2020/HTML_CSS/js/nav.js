$(document).ready(function() {

  let lastScrollTop = 0;
  $(window).scroll(function() {
    let st = $(this).scrollTop();
    let banner = $('#navbar');
    setTimeout(function() {
      if (st < lastScrollTop) {
        $(banner).animate({
          top: 0
        }, 10);
      } else {
        $(banner).animate({
          top: "-4rem"
        }, 10);
      }
      lastScrollTop = st;
    }, 10);
  });
});
