<?php
// define variables and set to empty values
$firstNameErr = $lastNameErr = $emailErr = $itemErr = $quantityErr = "*";
$fname = $lname = $email = $donation = $quantity = $item = $quantity = "";


if ($_SERVER["REQUEST_METHOD"] == "POST") {
  if (empty($_POST["fname"])) {
    $firstNameErr = "* First name is required";
  } else {

    // check if name only contains letters and whitespace
    if (!preg_match("/^[a-zA-Z-' ]*$/",$fname)) {
      $firstNameErr = "* Only letters and white space allowed";
    } else {
      $fname = test_input($_POST["fname"]);
      $firstNameErr = "";
    }
  }

  if (empty($_POST["lname"])) {
    $lastNameErr = "* Last name is required";
  } else {

    // check if name only contains letters and whitespace
    if (!preg_match("/^[a-zA-Z-' ]*$/",$lname)) {
      $lastNameErr = "* Only letters and white space allowed";
    } else {
      $lname = test_input($_POST["lname"]);
      $lastNameErr = "";
    }
  }

  if (empty($_POST["email"])) {
    $emailErr = "* Email is required";
  } else {

    // check if e-mail address is well-formed
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
      $emailErr = "* Invalid email format";
    } else {
      $email = test_input($_POST["email"]);
      $emailErr = "";
    }
  }
  if (empty($_POST["item"]) || $_POST["item"] == '--Select a Coral!--') {
    $itemErr = "* Coral is required for purchase";
  } else {
    $item = $_POST["item"];
    $itemErr = "";
  }

  if (empty($_POST["quantity"]) || $_POST["quantity"] < 0) {
    $quantityErr = "* Atleast one coral is required for purchase";
  } else {
    $quantity = $_POST["quantity"];
    $quantityErr = "";
  }
}

function test_input($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}

?>
<?php include 'Unit1_header.php';?>
<main>
  <div id="forms-container">
    <fieldset id="forms-container-fieldset">
      <legend id="forms-container-legend" style="margin-left: 1em; padding: 0.2em 0.8em ">Coral Order Form</legend>
      <form method="POST" action="Unit1_process_order.php">
        <!-- <?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?> -->
        <fieldset id="personal-info-fieldset">
          <legend id="personal-info-legend">Personal Information</legend>
            <label for="fname">First Name:</label>
            <input type="text" name="fname" value="<?php echo $fname;?>" required placeholder="Enter your first name" pattern="[A-Za-z ']*" title="Names can only include letters, spaces and apostrophe">
            <span class="error"><?php echo $firstNameErr;?></span>
          <br>
            <label for="lname">Last Name:</label>
            <input type="text" name="lname" value="<?php echo $lname;?>" required placeholder="Enter you last name" pattern="[A-Za-z ']*" title="Names can only include letters, spaces and apostrophe">
            <span class="error"><?php echo $lastNameErr;?></span>
          <br>
            <label for="email">E-mail:</label>
            <input type="email" name="email" value="<?php echo $email;?>" required placeholder="Enter your email" pattern="(.+)@(.+)(.+)">
            <span class="error" title="Enter a valid email address."><?php echo $emailErr;?></span>
          <br>
        </fieldset>

        <fieldset id="product-info-fieldset">
          <legend id="product-info-legend" style="margin-left: 1em; padding: 0.2em 0.8em ">Product Information</legend>
          <span class="error"><?php echo $itemErr;?></span>
          <div class="select">
              <select name="item" required oninvalid="this.setCustomValidity('You must select a coral first!')" oninput="this.setCustomValidity('')" title="Choose a coral first!" >
                  <option value="" selected disabled hidden>--Select a Coral!--</option>
                  <option value="Blizzard">Reef Raft USA "Blizzard" Signature Acro | 1/2" - 3/4" | $750.00</option>
                  <option value="Wicked Tuna">Reef Raft USA "Wicked Tuna" Signature Acro | 1/2" - 3/4" | $900.00</option>
                  <option value="Rising Sun">Reef Raft USA "Rising Sun" Signature Acro | 1/2" - 3/4" | $799.99</option>
                  <option value="Longhorn">Reef Raft USA "Longhorn" Signature Acro | 1/2" - 3/4" | $750.00</option>
                  <option value="Gonipora" style="background-image:url('images/Gonipora.jpg')">Reef Raft USA "Gonipora" | 1/2" - 3/4" | $350.00</option>
                  <option value="Holygrail">Reef Raft USA "Holygrail"| 1 head | $950.00</option>
                  <option value="Sleeping Beauty">Reef Raft USA "Sleeping Beauty" Favia | 1 eye | $125.00</option>
              </select>
              <div class="select_arrow"></div>

          </div>
          <label for="quantity">Quantity:</label>
          <input type="number" id="quantity" name="quantity" min="1" max="100" required oninvalid="this.setCustomValidity('You must purchase at least 1 coral!')" oninput="this.setCustomValidity('')"  title="You must purchase at least 1 coral!">
          <span class="error"><?php echo $quantityErr;?></span>

        </fieldset>
        <label for="purchase" id="donation-label">Round up to the nearest dollar for a donation?</label>
        <input id="donation-yes" type="radio" name="donation" value="yes">
        <label for="dontation-yes">Yes</label>
        <input id="donation-no" type="radio" name="donation" value="no"><label for="dontation-no"> No</label>
        <input id="purchase" type="submit" name="submit" value="Purchase">
      </form>
    </fieldset>
  </div>
  <div id="products-container">
    <fieldset id="products-container-fieldset">
      <legend id="products-container-legend">
        Select a coral to view it!
      </legend>
    </fieldset>
  </div>
</main>
<?php include 'Unit1_footer.php';?>
<script>
  $(document).ready(function() {

    // Show Product Images
    $("select").on("change", function() {
      let name = $(this).val();
      let coralImgPath = "./images/"+name.replace(/\s+/g, '')+".jpg";
      $("#products-container-fieldset").children(":not(legend)").remove();
      $("#products-container-fieldset").append(`<img src="${coralImgPath}" />`);
      $("#products-container-legend").html(name);
    });

    // Current page Change
    $(".nav-main").on("click", function() {
      $(".current-page").removeClass("current-page");
      $(this).addClass("current-page");
    });
  });
</script>
</body>
</html>
