(function($) {
  $.fn.removeClasses = function(classes) {
    return this.removeClass(classes.join(' '));
  };
  $.fn.switchify = function(config) {
    config = config || {};
    let prefix = config.prefix || 'range-';
    let onCls = prefix + (config.onCls || 'on');
    let offCls = prefix + (config.offCls || 'off');
    let unsetCls = prefix + (config.unsetCls || 'unset');
    let $self = this;
    return this.on('change', function(e) {
      let value = parseInt(this.value, 10);
      switch (value) {
        case 1:
          return $self.removeClasses([unsetCls, offCls]).addClass(onCls);
        case 2:
          return $self.removeClasses([onCls, offCls]).addClass(unsetCls);
        case 3:
          return $self.removeClasses([onCls, unsetCls]).addClass(offCls);
        default:
          return $self;
      }
    });
  };
})(jQuery);


$(document).ready(function() {
  $('#range-filter').switchify({
    onCls: 'active',
    offCls: 'passive',
    unsetCls: 'all'
  }).on('change', function(e) {
    let $self = $(this);
    if ($self.hasClass('range-active')) {
      $('#theme-type-text').text('Party');
      document.documentElement.setAttribute('data-theme', 'party');
      localStorage.setItem('theme', 'party'); //add this

    } else if ($self.hasClass('range-passive')) {
      $('#theme-type-text').text('Dark');
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('theme', 'dark'); //add this

    } else if ($self.hasClass('range-all')) {
      $('#theme-type-text').text('Light');
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('theme', 'light');

    } else $('#theme-type-text').text('Theme Error!');
  });

  const currentTheme = localStorage.getItem('theme') ? localStorage.getItem('theme') : null;
  if (currentTheme !== null) {
    // document.documentElement.setAttribute('data-theme', currentTheme);
    let $self = $("#range-filter");
    let prefix = "range-";
    let unsetCls = "unset";
    let onCls = "on";
    let offCls = "off";

    // Add party themes here
    if (currentTheme == "party") {
      $self.removeClasses([unsetCls, offCls]).addClass(onCls);
      $('#theme-type-text').text('Party');
      document.documentElement.setAttribute('data-theme', 'party');
      localStorage.setItem('theme', 'party'); //add this
      $self.val(1)
    } else if (currentTheme == "dark") {
      $self.removeClasses([onCls, offCls]).addClass(unsetCls)
      $('#theme-type-text').text('Dark');
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('theme', 'dark'); //add this
      $self.val(3)
    } else if (currentTheme == "dark") {
      $self.removeClasses([onCls, unsetCls]).addClass(offCls);
      $('#theme-type-text').text('Light');
      document.documentElement.setAttribute('data-theme', 'light');
      localStorage.setItem('theme', 'light');
      $self.val(2)
    }
  }
});
