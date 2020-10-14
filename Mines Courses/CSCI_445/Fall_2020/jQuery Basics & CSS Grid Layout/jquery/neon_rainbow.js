/*jshint esversion: 8 */

// FOR NEON TEXT
// textColor: regular text fill color
// textSize: string on font size (e.g. "40px", "1rem", etc.)
// neonHighlight: highlight color for the neon
// neonHighlightColor: primary neon color
// neonHighlightHover: neon highlight color on hover
// neonFontHover: color of font on hover

// FOR NEON BORDER
// neonHighlight: highlight color (secondary white),
// neonHighlightColor: primary Color,
// neonHighlightHover: primary color on hover
// neonHighlightColorHover: 'hihglight color on hover,
// neonSpreadFactor: light spread factor
// hover: true/false; whether to have hover animations
// neonHoverSpreadFactor: Light spread factor on hover,
// hoverAnimationClasses: class to add on during hovering
// inset: true/false; whether the border should be inset also
// neonInsetSpreadFactor: Light spread factor for inset border
// neonInsetSpreadFactorHover: Light spread factor on hover for inset border


$.fn.neonText = function(options) {
  options = $.extend({
    textColor: '#FFFFFF',
    textSize: '40pt',
    neonHighlight: '#FFFFFF',
    neonHighlightColor: '#FF00DE',
    neonHighlightHover: '#00FFFF',
    neonFontHover: '#FFFFFF',
    neonSpreadFactor: 1
  }, options);
  return this.each(function() {
    $(this).css('color', options.textColor)
      .css('font-size', options.textSize)
      .css('text-shadow', '0 0 ' + 10 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 20 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 30 * options.neonSpreadFactor + 'px ' + options
        .neonHighlight + ',0 0 ' + 40 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 70 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 80 * options.neonSpreadFactor + 'px ' + options
        .neonHighlightColor + ',0 0 ' + 100 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor)
      .hover(
        function() {
          $(this)
            .css('text-shadow', '0 0 ' + 10 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 20 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 30 * options.neonSpreadFactor + 'px ' + options
              .neonHighlight + ',0 0 ' + 40 * options.neonSpreadFactor + 'px ' + options.neonHighlightHover + ',0 0 ' + 70 * options.neonSpreadFactor + 'px ' + options.neonHighlightHover + ',0 0 ' + 80 * options.neonSpreadFactor + 'px ' +
              options.neonHighlightHover + ',0 0 ' + 100 * options.neonSpreadFactor + 'px ' + options.neonHighlightHover)
            .css('color', options.neonFontHover);
        },
        function() {
          $(this)
            .css('color', options.textColor)
            .css('text-shadow', '0 0 ' + 10 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 20 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 30 * options.neonSpreadFactor + 'px ' + options
              .neonHighlightColor + ',0 0 ' + 40 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 70 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 80 * options.neonSpreadFactor +
              'px ' + options.neonHighlightColor + ',0 0 ' + 100 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor);
        }
      );
  });
};

$.fn.neonBorder = function(options) {
  options = $.extend({
    neonHighlight: '#FFFFFF',
    neonHighlightColor: '#FF00DE',
    neonHighlightHover: '#00FFFF',
    neonHighlightColorHover: '#FFFFFF',
    neonSpreadFactor: 1,
    neonHoverSpreadFactor: 1,
    hoverAnimationClasses: null,
    inset: true,
    neonInsetSpreadFactor: 1,
    neonHoverInsetSpreadFactor: 1,
    hover: true
  }, options);

  let border = '0 0 ' + 10 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 20 * options.neonSpreadFactor + 'px ' + options.neonHighlight + ',0 0 ' + 30 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor +
    ',0 0 ' + 40 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 70 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor + ',0 0 ' + 80 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor +
    ',0 0 ' + 100 * options.neonSpreadFactor + 'px ' + options.neonHighlightColor;
  if (options.inset == true) {
    border += ',inset 0 0 ' + options.neonInsetSpreadFactor + 'px ' + options.neonHighlight + ',inset 0 0 ' + 2 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlight + ',inset 0 0 ' + 10 * options.neonInsetSpreadFactor + 'px ' +
      options.neonHighlight + ',inset 0 0 ' + 10 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 20 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 30 * options
      .neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 40 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor;
  }

  let borderHover = '0 0 ' + 10 * options.neonHoverSpreadFactor + 'px ' + options.neonHighlightHover + ',0 0 ' + 20 * options.neonHoverSpreadFactor + 'px ' + options.neonHighlightHover + ',0 0 ' + 30 * options.neonHoverSpreadFactor + 'px ' +
    options.neonHighlightHover + ',0 0 ' + 40 * options.neonHoverSpreadFactor + 'px ' + options.neonHighlightColorHover + ',0 0 ' + 70 * options.neonHoverSpreadFactor + 'px ' + options.neonHighlightColorHover + ',0 0 ' + 80 * options
    .neonHoverSpreadFactor + 'px ' + options.neonHighlightColorHover + ',0 0 ' + 100 * options.neonHoverSpreadFactor + 'px ' + options.neonHighlightColorHover + ', inset 0 0 ' + 100 * options.neonHoverSpreadFactor + 'px ' + options
    .neonHighlighColorHover;

  if (options.inset == true) {
    border += ',inset 0 0 ' + options.neonInsetSpreadFactorHover + 'px ' + options.neonHighlight + ',inset 0 0 ' + 2 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlight + ',inset 0 0 ' + 10 * options.neonInsetSpreadFactor + 'px ' +
      options.neonHighlight + ',inset 0 0 ' + 10 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 20 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 30 * options
      .neonInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 40 * options.neonInsetSpreadFactor + 'px ' + options.neonHighlightColor;
    borderHover += ',inset 0 0 ' + options.neonHoverInsetSpreadFactorHover + 'px ' + options.neonHighlight + ',inset 0 0 ' + 2 * options.neonHoverInsetSpreadFactor + 'px ' + options.neonHighlight + ',inset 0 0 ' + 10 * options
      .neonHoverInsetSpreadFactor + 'px ' + options.neonHighlight + ',inset 0 0 ' + 10 * options.neonHoverInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 20 * options.neonHoverInsetSpreadFactor + 'px ' + options
      .neonHighlightColor + ',inset 0 0 ' + 30 * options.neonHoverInsetSpreadFactor + 'px ' + options.neonHighlightColor + ',inset 0 0 ' + 40 * options.neonHoverInsetSpreadFactor + 'px ' + options.neonHighlightColor;
  }


  return this.each(function() {
    $(this).css({
      'boxShadow': border
    });
    $(this).stop().animate({
      boxShadow: border
    });
  }).hover(
    function() {
      if (options.hover) {
        $(this).stop().animate({
          boxShadow: borderHover
        });
        if (options.hoverAnimationClasses != null) $(this).toggleClass(options.hoverAnimationClasses);
      }
    },
    function() {
      if (options.hover) {
        $(this).stop().animate({
          boxShadow: border
        });
        if (options.hoverAnimationClasses != null) $(this).toggleClass(options.hoverAnimationClasses);
      }
    }
  );

};

$.fn.rainbowText = function(options) {
  options = $.extend({
    textColor: '#FFFFFF',
    textSize: '',
    neonHighlight: '#FFFFFF',
    neonHighlightColor: '#FF00DE',
    neonHighlightHover: '#00FFFF',
    neonFontHover: '#FFFFFF',
    neonSpreadFactor: 1
  }, options);
  let svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttributeNS(null, 'id', 'rainbow__svg');
  svg.setAttributeNS(null, 'display', 'none');
  $("body").append(svg);
  let defs = document.createElementNS('xmlns', "defs");
  svg.appendChild(defs);
  $("#rainbow__svg > defs").html(`
  <linearGradient id="rainbow__gradient" gradientTransform="rotate(90)">
    <stop offset="20%" stop-color="violet" />
    <stop offset="90%" stop-color="indigo" />
    <stop offset="20%" stop-color="blue" />
    <stop offset="90%" stop-color="green" />
    <stop offset="20%" stop-color="yellow" />
    <stop offset="90%" stop-color="orange" />
    <stop offset="20%" stop-color="red" />
  </linearGradient>
  `);
  $(".intro-title-text").css({
    '-webkit-text-stroke': 'url(#rainbow__gradient)'
  });
  return this.each(function() {
    $(".intro-title-text").css({
      '-webkit-text-stroke': ' 0.35px url(#rainbow__gradient)'
    });
  });
  // .hover(
  // function () {
  // 	$(this).();
  // },
  // function () {
  // 	$(this).();
  // }
  // );
};

$.fn.rainbowNeonText = function(options) {
  options = $.extend({
    textSize: '1em',
    neonSpreadFactor: 1,
    child: 'a'
  }, options);

  return this.each(function() {
    let text = $(this).find(options.child).first().text();
    let svgText =
      `
        <svg class="rainbow" width="${$(this).width()}" height="${$(this).height()}" viewBox="0 0 ${$(this).width()} ${$(this).height()}">
          <title>${text}</title>
          <g style="overflow:visible; text-anchor: middle;stroke-width: 0.25px; stroke-linecap='round'; font-size: ${options.textSize};  fill: none;">
            <defs>
              <linearGradient id="rainbow__gradient">
                <stop offset="12.5%" stop-color="violet" />
                <stop offset="25%" stop-color="rgb(79, 8, 181)" />
                <stop offset="37.5%" stop-color="rgb(86, 111, 228)" />
                <stop offset="50%" stop-color="rgb(29, 148, 164)" />
                <stop offset="62.5%" stop-color="green" />
                <stop offset="75%" stop-color="yellow" />
                <stop offset="87.5%" stop-color="orange" />
                <stop offset="100%" stop-color="red" />
              </linearGradient x2="10em" y2="1em">
              <filter id="glow1" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="${options.neonSpreadFactor*10} ${options.neonSpreadFactor*10}" result="glow"/>
                <feMerge>
                  <feMergeNode in="glow"/>
                  <feMergeNode in="glow"/>
                  <feMergeNode in="glow"/>
                </feMerge>
              </filter>
              <filter id="glow2" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="${options.neonSpreadFactor*4} ${options.neonSpreadFactor*4}" result="glow"/>
                <feMerge>
                  <feMergeNode in="glow"/>
                  <feMergeNode in="glow"/>
                </feMerge>
              </filter>
              <filter id="glow3" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="${options.neonSpreadFactor*2} ${options.neonSpreadFactor*2}" result="glow"/>
              </filter>
              <mask id="textMask">
                <text style=" fill: none;" x="${$(this).width()/2}" y="${$(this).height()/2}">
                  ${text}
                </text>
              </mask>
            </defs>
            <text style="filter: url(#glow1);" x="${$(this).width()/2}" y="${$(this).height()/2}">
              ${text}
            </text>
            <text style="overflow:visible; filter: url(#glow2);" x="${$(this).width()/2}" y="${$(this).height()/2}">
              ${text}
            </text>
            <text x="${$(this).width()/2}" y="${$(this).height()/2}" style="overflow:visible; stroke: url(#rainbow__gradient);">
              ${text}
            </text>
            <g mask="url(#textMask)"">
              <g style="filter: url(#glow3); fill: url(#rainbow__gradient)">
                <text style="fill: none;" x="${$(this).width()/2}" y="${$(this).height()/2}">
                  ${text}
                </text>
              </g>
            </g>
          </g>
        </svg>
        `;
    $(this).html(svgText);
  });
  // .hover(
  // function () {
  // 	$(this).();
  // },
  // function () {
  // 	$(this).();
  // }
  // );
};


$.fn.rainbowTextOutline = function(options) {
  options = $.extend({
    textSize: '1em',
    neonSpreadFactor: 1,
    child: 'a',
    color: "none",
    width: "0.25px"
  }, options);

  return this.each(function() {
    let text = $(this).find(options.child).first().text();
    let svgText =
      `
        <svg class="rainbow" width="${$(this).width()}" height="${$(this).height()}" viewBox="0 0 ${$(this).width()} ${$(this).height()}">
          <title>${text}</title>
          <g style="overflow:visible; text-anchor: middle;stroke-width: ${options.width}; stroke-linecap='round'; font-size: ${options.textSize}; fill: ${options.color}">
          <defs>
            <linearGradient id="rainbow__gradient">
              <stop offset="12.5%" stop-color="violet" />
              <stop offset="25%" stop-color="rgb(79, 8, 181)" />
              <stop offset="37.5%" stop-color="rgb(86, 111, 228)" />
              <stop offset="50%" stop-color="rgb(29, 148, 164)" />
              <stop offset="62.5%" stop-color="green" />
              <stop offset="75%" stop-color="yellow" />
              <stop offset="87.5%" stop-color="orange" />
              <stop offset="100%" stop-color="red" />
            </linearGradient x2="10em" y2="1em">
            <filter id="glow1" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="${options.neonSpreadFactor*10} ${options.neonSpreadFactor*10}" result="glow"/>
              <feMerge>
                <feMergeNode in="glow"/>
                <feMergeNode in="glow"/>
                <feMergeNode in="glow"/>
              </feMerge>
            </filter>
            <filter id="glow2" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="${options.neonSpreadFactor*4} ${options.neonSpreadFactor*4}" result="glow"/>
              <feMerge>
                <feMergeNode in="glow"/>
                <feMergeNode in="glow"/>
              </feMerge>
            </filter>
            <filter id="glow3" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="${options.neonSpreadFactor*2} ${options.neonSpreadFactor*2}" result="glow"/>
            </filter>
              <mask id="textMask">
                <text style="fill: none;" x="${$(this).width()/2}" y="${$(this).height()/2}">
                  ${text}
                </text>
              </mask>
            </defs>
            <text style="filter: url(#glow1);" x="${$(this).width()/2}" y="${$(this).height()/2}">
              ${text}
            </text>
            <text style="filter: url(#glow2);" x="${$(this).width()/2}" y="${$(this).height()/2}">
              ${text}
            </text>
            <text x="${$(this).width()/2}" y="${$(this).height()/2}" style="stroke: url(#rainbow__gradient); ">
              ${text}
            </text>
            <g mask="url(#textMask)">
              <g style="filter: url(#glow3);fill: url(#rainbow__gradient)">
                <text style="fill: none;" x="${$(this).width()/2}" y="${$(this).height()/2}">
                  ${text}
                </text>
              </g>
            </g>
          </g>
        </svg>
        `;
    $(this).html(svgText);
  });
  // .hover(
  // function () {
  // 	$(this).();
  // },
  // function () {
  // 	$(this).();
  // }
  // );
};

$.fn.rainbowBorder = function(options) {
  options = $.extend({
    width: "0.25px"
  }, options);

  return this.each(function() {
    $(this).css({
      'border': `${options.width} solid`,
      'borderImage': ` linearGradient(violet 15%, indigo 30%, blue 45%, green 60%, yellow 70%, orange 85%, red 100%)`
    });
  });
  // .hover(
  // function () {
  // 	$(this).();
  // },
  // function () {
  // 	$(this).();
  // }
  // );
};
