<html>
  <head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
	<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">

	<style>
#table-wrapper {
  position:relative;
}
#table-scroll {
  height:550px;
  overflow:auto;  
  margin-top:20px;
}
#table-wrapper table {
  width:100%;

}
#table-wrapper table * {
  background:yellow;
  color:black;
}
#table-wrapper table thead th .text {
  position:absolute;   
  top:-20px;
  z-index:2;
  height:20px;
  width:35%;
  border:1px solid red;
  
}
table tr td:empty { 
  width: 50px;
}

.audusd{ background: #2196f3; }
.euraud{ background: #3f51b5; }
.gbpusd{ background: #673ab7; }
.eurusd{ background: #673ab7; }

</style>
<script>
function myalert() {
  alert("Please select the Currency and Frequency in order to continue.");
}
</script>
<script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>  
   <script type="text/javascript">
$(document).ready(function(){
    $("select[name='chooseCurrency']").change(function(){
        $(this).find("option:selected").each(function(){
            var optionValue = $(this).attr("value");
            if(optionValue){
                $(".currency").not("." + optionValue).hide();
                $("." + optionValue).show();
            } else{
                $(".currency").hide();
            }
        });
    }).change();
});
        </script>
<script>
function ddselect()
{
var d = document.getElementById("t1");
var displaytext= d.options[d.selectedIndex].text;
document.getElementById("textvalue").innerHTML=displaytext;
}

function ddselect1()
{
var d = document.getElementById("t2");
var displaytext= d.options[d.selectedIndex].text;
document.getElementById("textvalue1").innerHTML=displaytext;
}

function dbselect()
{
	ddselect();
	ddselect1();
}
</script>		


 <link rel="stylesheet" href="css/style.css"> 
</head>     

<body  onload = "myalert()">
    <header>
      <div class="container">
        <div id="branding">
        <h1 style ="font-size:36; margin-bottom: 15px;"><span style ="font-size:36" class="highlight" st>Strategy</span> Two</h1>
        </div>
    </header>
<?php



if ($_SERVER["REQUEST_METHOD"]=="POST")
{



	$servername = "localhost:3308";
	$username = "robotrade";
	$password = "robotrade";
	$dbname = "final2";


	$conn = new mysqli($servername, $username, $password, $dbname);
	
	if($conn->connect_error) 
	{
		die("Connection failed: " . $conn->connect_error);
	}
	

	if ($sum ="")
	{
		$sum="audusd4";
	}

	$name = $_POST["table"];	
	$freq = $_POST["frequency"];
	$sum =$name.$freq;
	if ($name ="")
	{
		$sum="audusd4";
	}

$sql = "SELECT *FROM ".$sum;
$result = $conn->query($sql);
}
?>




<form method = "post">
<div style = "margin-left:25px; margin-top: 10px">
        <b>Currency:</b><select id = "t1" name="table" class="browser-default custom-select custom-select-lg mb-3" style= "width:auto" onchange= "ddselect()">
			<option value="audusd">AUD/USD</option>
			<option value="euraud">EUR/AUD</option>
			<option value="eurusd">EUR/USD</option>
			<option value="gbpusd">GBP/USD</option>
            </select>
		<b>Frequency:</b><select id = "t2" name="frequency" class="browser-default custom-select custom-select-lg mb-3" style= "width:auto;" onchange = "ddselect1()";>
            <option value="h4">4 hours</option>
            <option value="h1">1 hours</option>
            </select>
		<input type = "submit" name ="submit" value = "submit" onclick= "dbselect()" class="btn btn-primary btn-lg mb-3" style= "width:auto;">
			<a href="index.php"><button type="button" class="btn btn-lg btn-link" style= "width:auto; float:right;margin-right:25px">Home</button></a>
		<a href="strategy1.php"><button type="button" class="btn btn-lg btn-link" style= "width:auto; float:right;margin-right:25px">Strategy One</button></a>	
		<select name="chooseCurrency" id="chooseCurrency" onchange="show()" class="browser-default custom-select custom-select-lg mb-3" style= "width:auto; float:right;margin-right:25px">
			<option value="choose_currency">Choose currency</option>           
			<option value="AUDUSD">AUDUSD</option>
            <option value="EURAUD">EURAUD</option>
            <option value="GBPUSD">GBPUSD</option>
			<option value="EURUSD">EURUSD</option>
        </select>

</div>
		</form>






<p style="margin-left:25px;display:inline; font-weight: bold;">Currency currently selected:</p>
		<p id="textvalue" style="display:inline;"></p>
		<br>


<p style="margin-left:25px;display:inline; font-weight: bold;">Frequency currently selected:</p>
		<p id="textvalue1" style="display:inline;"></p>
<center>
	<table>
	<td>
	<div class = "first table" id = final3 >
	<div id ="table-scroll">
    <table class = "table table-striped">
	  <thead class="thead-dark">
    <tr>
      <th scope="col">Entry Date</th>
      <th scope="col">Buy/Sell</th>
      <th scope="col">Entry Price</th>
      <th scope="col">Exit Date</th>
	  <th scope="col">Exit Price</th>
	  <th scope="col">P/L</th>
	  <th scope="col">Comment</th>
    </tr>
  </thead>
  <tbody>
<?php
	if ($result->num_rows > 0)
    {
        while($row = $result->fetch_assoc()){
            echo "<tr><td>" . $row["entrydate"] ."</td><td>" . $row["bs"] . "</td><td>" . $row["entryprice"]."</td><td>" . $row["exitdate"] . "</td><td>" .$row["exitprice"]. "</td> <td>" . $row["pnl"] . "</td> <td>" . $row["comment"] . "</td></tr>";
        }
    }      
        else{ echo "0 reults";
	}

$conn->close();                     
?>
</tbody>
</table>
</div>
</div>
</td>
<td></td>
<div>
<td>
    <div class="audusd currency" id = AUDUSD>
	<a href="https://plotly.com/~suvir6/180/?share_key=3mPWQ1MuMPH4FDDpc2JOrL" target="_blank" title="AUDUSD 4 Hour data" style="display: block; text-align: center;"><img src="https://plotly.com/~suvir6/180.png?share_key=3mPWQ1MuMPH4FDDpc2JOrL" alt="AUDUSD 4 Hour data" style="max-width: 100%;width: 600px; height:550px; "  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="suvir6:180" sharekey-plotly="3mPWQ1MuMPH4FDDpc2JOrL" src="https://plotly.com/embed.js" async></script>
	</div>
	<div class="euraud currency" id = EURAUD>
    <a href="https://plotly.com/~suvir6/178/?share_key=FIJEQEooCphGkk24uATaJK" target="_blank" title="EURAUD 4 Hour data" style="display: block; text-align: center;"><img src="https://plotly.com/~suvir6/178.png?share_key=FIJEQEooCphGkk24uATaJK" alt="EURAUD 4 Hour data" style="max-width: 100%;width: 600px; height:550px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="suvir6:178" sharekey-plotly="FIJEQEooCphGkk24uATaJK" src="https://plotly.com/embed.js" async></script>
	</div>
	<div class="gbpusd currency" id = GBPUSD>
	<a href="https://plotly.com/~suvir6/173/?share_key=AO054g8zyJfvxR7vIDJnIs" target="_blank" title="GBPUSD 4 Hour data" style="display: block; text-align: center;"><img src="https://plotly.com/~suvir6/173.png?share_key=AO054g8zyJfvxR7vIDJnIs" alt="GBPUSD 4 Hour data" style="max-width: 100%;width: 600px; height:550px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="suvir6:173" sharekey-plotly="AO054g8zyJfvxR7vIDJnIs" src="https://plotly.com/embed.js" async></script>
	</div>
	<div class="eurusd currency" id = EURUSD>
    <a href="https://plotly.com/~suvir6/171/?share_key=T2RgTDYcCvKyroXwA62qds" target="_blank" title="EURUSD 4 Hour data" style="display: block; text-align: center;"><img src="https://plotly.com/~suvir6/171.png?share_key=T2RgTDYcCvKyroXwA62qds" alt="EURUSD 4 Hour data" style="max-width: 100%;width: 600px; height:550px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="suvir6:171" sharekey-plotly="T2RgTDYcCvKyroXwA62qds" src="https://plotly.com/embed.js" async></script>
	</div>
</td>
</div>
</table>
</center>
<footer>
  <p>A University of Sydney project by <b>Group 2</b>
</footer>
    </body>    
</html>