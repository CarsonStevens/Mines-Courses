/*jshint esversion: 8 */

$(document).ready(function() {
  let typers = [$(".typewriter")];
  let typersOriginals = [];
  typers.forEach((typer, i) => {
    typersOriginals[i] = $(typer).html();
  });
  let currentWriter = 0;
  let transitionTime = 2000;
  let currentlyWriting = false;
  rotateTypewriter();

  // Observable Trigger to start typewriter
  // $('').observe({
  //   print: true,
  //   listeners:  [{type : "attributes",
  //               attribute: "class",
  //               listeners : "flipster__item--current"}],
  //   config: {
  //     childList: false,
  //     characterData: false,
  //     subtree: false,
  //     attributes: true
  //   },
  //   onListenerTrue: rotateTypewriter
  // });

  function rotateTypewriter() {
    if (!currentlyWriting) {
      currentlyWriting = true;
      $(".typewriter-container").css({
        'display': 'none'
      });
      $(typers[currentWriter]).parent().css({
        'display': 'grid'
      });

      let typewriter = setupTypewriter($(typers[currentWriter]).get(0));
      typewriter.type();
      currentWriter++;
      if (currentWriter >= typers.length) currentWriter = 0;
    }
  }

  function strToObj(str) {
    let obj = {};
    if (str && typeof str === 'string') {
      eval("obj =" + str);
    }
    return obj;
  }

  function setupTypewriter(t) {
    $(t).children(".code-output-container").remove();
    let HTML = typersOriginals[currentWriter];

    t.innerHTML = "";

    let cursorPosition = 0,
      tag = "",
      writingTag = false,
      tagOpen = false,
      typeSpeed = 15,
      tempTypeSpeed = 0;

    let output = function() {
      let output = strToObj($(t).data("output"));
      let outputTerms = output.terms || [];
      let spacing = output.timeSpacing || 0;
      let delay = output.delay || 0;
      let animationClasses = output.animationClasses || '';
      $(t).append("<br><span class='code-output-container'><span class='code-output-container-title'></span></span>");
      setTimeout(function() {
        for (let i = 0; i < outputTerms.length; i++) {
          setTimeout(function() {
            $(t).children(".code-output-container").append(`<span class="code-output ${animationClasses}"> ${outputTerms[i]}</span>`);
          }, i * spacing);

        }
      }, delay);

      setTimeout(function() {
        currentlyWriting = false;
        rotateTypewriter();
      }, transitionTime + outputTerms.length * spacing);
    };

    let type = function() {

      if (writingTag === true) {
        tag += HTML[cursorPosition];
      }

      if (HTML[cursorPosition] === "<") {
        tempTypeSpeed = 0;
        if (tagOpen) {
          tagOpen = false;
          writingTag = true;
        } else {
          tag = "";
          tagOpen = true;
          writingTag = true;
          tag += HTML[cursorPosition];
        }
      }
      if (!writingTag && tagOpen) {
        tag.innerHTML += HTML[cursorPosition];
      }
      if (!writingTag && !tagOpen) {
        if (HTML[cursorPosition] === " ") {
          tempTypeSpeed = 0;
        } else {
          tempTypeSpeed = (Math.random() * typeSpeed) + 50;
        }
        t.innerHTML += HTML[cursorPosition];
      }
      if (writingTag === true && HTML[cursorPosition] === ">") {
        tempTypeSpeed = (Math.random() * typeSpeed) + 20;
        writingTag = false;
        if (tagOpen) {
          let newSpan = document.createElement("span");
          t.appendChild(newSpan);
          $(newSpan).html(tag);
          tag = newSpan.firstChild;
        }
      }

      cursorPosition += 1;
      if (cursorPosition < HTML.length - 1) {
        setTimeout(type, tempTypeSpeed);
      }
      if (cursorPosition >= HTML.length - 1) {
        return output();
      }
    };

    return {
      type: type
    };
  }
});
