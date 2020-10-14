$(document).ready(function() {
  let lastScrollTop = 0;
  $(window).scroll(function() {
    let st = $(this).scrollTop();
    let banner = $('#navbar');
    setTimeout(function() {
      if (st < lastScrollTop && $("#hamburger").css('display') == "none") {
        $(banner).animate({
          top: 0
        }, 10);
      } else {
        $("#hamburger .ham").removeClass('active');
        $(banner).animate({
          top: "-100%"
        }, 10);
      }
      lastScrollTop = st;
    }, 10);
  });

  $("#hamburger .ham").on("click", function() {
    this.classList.toggle('active');
    if ($(this).hasClass("active")) {
      $("#navbar").animate({
        top: 0
      }, 10);
    } else {
      $("#navbar").animate({
        top: "-100%"
      }, 10);
    }
  });

  const hamburgerSetup = (function() {
    if ($("#hamburger").css('display') != "none"){
        $("#navbar").animate({
          top: "-100%"
        }, 10);
      } else {
        $(this).addClass("active");
        $("#navbar").animate({
          top: 0
        }, 10);
      }
    });
  hamburgerSetup();
  window.addEventListener("resize",hamburgerSetup);

});
