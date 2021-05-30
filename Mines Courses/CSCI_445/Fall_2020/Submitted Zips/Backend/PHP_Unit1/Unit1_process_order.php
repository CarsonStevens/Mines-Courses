  <?php include 'Unit1_header.php';?>
  <?php

  function  apply_tax($price) {
    // Tax rate of 7.65%
    return $price * 1.0765;
  }
  $coral_directory = array("Blizzard"=>"750.00", "Wicked Tuna"=>"900.00", "Rising Sun"=>"799.99", "Longhorn"=>"750.00", "Sleeping Beauty"=>"125.00", "Holygrail"=>"950.00", "Gonipora"=>"350.00");
  $fname = $_POST['fname'];
  $lname = $_POST['lname'];
  $email = $_POST['email'];
  $quantity = $_POST['quantity'];
  $item = $_POST["item"];
  $total = $subtotal = $total_with_tax = 0;
  $donation = "no";

  if (!empty($_POST["item"]) && !empty($_POST["quantity"])) {
    $subtotal = $coral_directory[$item] * $quantity;
    $total_with_tax = apply_tax($subtotal);

  }

  if (!empty($_POST["donation"]) && ($_POST["donation"] == 'yes')) {
    $donation = "Thank you so much for your donation!";
    $total = ceil($total_with_tax);
  } else {
    $donation = "";
    $total = $total_with_tax;
  }

  echo "<section id='receipt'><span>Thank you  for your order, " . $fname . " " . $lname . " (" . $email . ")" . "</span>" . "<br />";
  echo "You purchased " . $quantity . " " . $item . "<br />";
  echo "Subtotal: $" . number_format($subtotal, 2, '.', '') . "<br />";
  echo "Total including tax: $" . number_format($total_with_tax, 2, '.', '') . "<br />";
  echo "Total with donation: $" . number_format($total, 2, '.', '') . "<br />";
  echo $donation . "</section>";
  ?>
  <?php include 'Unit1_footer.php';?>
</body>
</html>
