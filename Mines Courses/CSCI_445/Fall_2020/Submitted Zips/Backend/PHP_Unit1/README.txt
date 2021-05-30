1. Cross-site scripting (XSS) is a vulnerability that enables hackers to inject
client-side script into web pages. Explain the potential issue with using
$_SERVER["PHP_SELF"] as the form action, and how to avoid that issue.

  The potential danger is that an attacker could inject a script as input and without checking if it is sanitized, it could then execute the attacker's script. The methods to avoid that issue are to use Regex pattern matching and to sanitize the inputs with htmlspecialchars() to replace "<>" and other chars with HTML code equivalents. Without these characters, it is much harder to inject a script. Other forms of validation might be specific to what the input is.

2. Explain why it's important to have server-side validation, and why you might
want both client- and server-side.

  Server-side validation is important to make sure that you don't end up injecting harmful input into whatever you are doing with the data. The regex pattern matching solves some problems, but without properly sanitizing the input, things like SQL injection scripts could tamper with your database. It's good to have client-side validation also as it prevents a request to the severs giving the user more instant feedback and in the same way protects the server by first checking on the client side.
