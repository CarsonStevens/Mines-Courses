Author: Carson Stevens

What styling is used to make the links display horizontally (i.e., on larger screens)

  Their display is done horizontally with float and displaying each <a> as a block. When triggered for a smaller screen, the float is removed making each take the full width.

  I used a flex box with the flex direction set to row or col which was triggered by a media query.

What part of the code (in the HTML file) actually causes the hamburger icon to appear? Be specific.

.topnav .icon {
  display: none;
}

@media screen and (max-width: 600px) {
  .topnav a:not(:first-child) {display: none;}
  .topnav a.icon {
    float: right;
    display: block;
  }
}

  My hamburger is an svg that has a display setting that is toggled between block and none triggered by a media query. Both our hamburger icon have an active/responsive class that is added and removed on click to go from block to none. In mine, this also triggers the transformation between hamburger and X. Besides that, when the screen is small and it is showing, the menu is fixed at the top. Clicking it changes the top value to let it not show or show.

How does this code use CSS pseudo-classes to display only home and the hamburger when the screen size is small?

  The code uses a:not(:first-child){display:none} in a media query to toggle showing and not showing the list items. All a tags that aren't the 'Home' are set to display none when the screen is small. This line:
  .topnav a.icon {
    float: right;
    display: block;
  }
  inside the media query makes the icon show.
