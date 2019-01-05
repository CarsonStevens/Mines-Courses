// Author: Carson Stevens

public class JavaBatIndividual {

	// tenRun exercise
	public int[] tenRun(int[] nums) {
	  for(int i = 0; i < nums.length; i++) {
		  if(nums[i]%10 == 0) {	//Ignores all numbers until it finds a multiple of ten
			  /*
			   * Checks to make sure the next number is index able and that it isn't another 
			   * multiple of 10. If it is, then it breaks to change to that number.
			   */
			  for(int j = i + 1; !(nums[j] % 10 == 0) && (j < nums.length); j++){ 
				  nums[j] = nums[i];
			  }
		  }
	  }
	  return nums;
	}
/*
Expected	Run		
tenRun([2, 10, 3, 4, 20, 5]) → [2, 10, 10, 10, 20, 20]	[2, 10, 10, 10, 20, 20]	OK	
tenRun([10, 1, 20, 2]) → [10, 10, 20, 20]	[10, 10, 20, 20]	OK	
tenRun([10, 1, 9, 20]) → [10, 10, 10, 20]	[10, 10, 10, 20]	OK	
tenRun([1, 2, 50, 1]) → [1, 2, 50, 50]	[1, 2, 50, 50]	OK	
tenRun([1, 20, 50, 1]) → [1, 20, 50, 50]	[1, 20, 50, 50]	OK	
tenRun([10, 10]) → [10, 10]	[10, 10]	OK	
tenRun([10, 2]) → [10, 10]	[10, 10]	OK	
tenRun([0, 2]) → [0, 0]	[0, 0]	OK	
tenRun([1, 2]) → [1, 2]	[1, 2]	OK	
tenRun([1]) → [1]	[1]	OK	
tenRun([]) → []	[]	OK	
other tests
OK	

All Correct
*/
	
	// scoresIncreasing exercise
	public boolean scoresIncreasing(int[] scores) {
		  boolean answer = true;
		  int i = 0;
		  while(i < scores.length-1){
		    if(scores[i] <= scores[i+1]){ // Checks to see if the next number is larger
		      answer = true;
		    }
		    else{ // if it isn't larger, return false
		      answer = false;
		      return answer;
		    }
		    i++;
		  }
		  return answer;
		}
/*
Expected	Run		
scoresIncreasing([1, 3, 4]) → true	true	OK	
scoresIncreasing([1, 3, 2]) → false	false	OK	
scoresIncreasing([1, 1, 4]) → true	true	OK	
scoresIncreasing([1, 1, 2, 4, 4, 7]) → true	true	OK	
scoresIncreasing([1, 1, 2, 4, 3, 7]) → false	false	OK	
scoresIncreasing([-5, 4, 11]) → true	true	OK	

All Correct
 */
	
	// repeatEnd
	public String repeatEnd(String str, int n) {
	  String repeat = str.substring(str.length() - n); // Create String with ending characters wanted
	  String answer = "";
	  for(int i = 0; i < n; i++){	// Repeat the ending 'n' times
	    answer += repeat;
	  }
	  return answer;
	}

/*
Expected	Run		
repeatEnd("Hello", 3) → "llollollo"	"llollollo"	OK	
repeatEnd("Hello", 2) → "lolo"	"lolo"	OK	
repeatEnd("Hello", 1) → "o"	"o"	OK	
repeatEnd("Hello", 0) → ""	""	OK	
repeatEnd("abc", 3) → "abcabcabc"	"abcabcabc"	OK	
repeatEnd("1234", 2) → "3434"	"3434"	OK	
repeatEnd("1234", 3) → "234234234"	"234234234"	OK	
repeatEnd("", 0) → ""	""	OK	
other tests
OK	

All Correct
 */
	
	// canBalance exercise
	public boolean canBalance(int[] nums) {
	  int front = 0;
	  /*
	   * Iterates through a front and back changing the size of the front and back testing 
	   * each combo to see if its an even split.
	   */
	  for (int i = 0; i < nums.length; i++) {
	    front += nums[i];
	    int back = 0;
	    for (int j = nums.length-1; j > i; j--) {
	      back += nums[j];
	    }
	    if (back == front)
	      return true;
	  }
	  return false;
	}
/*
Expected	Run		
canBalance([1, 1, 1, 2, 1]) → true	true	OK	
canBalance([2, 1, 1, 2, 1]) → false	false	OK	
canBalance([10, 10]) → true	true	OK	
canBalance([10, 0, 1, -1, 10]) → true	true	OK	
canBalance([1, 1, 1, 1, 4]) → true	true	OK	
canBalance([2, 1, 1, 1, 4]) → false	false	OK	
canBalance([2, 3, 4, 1, 2]) → false	false	OK	
canBalance([1, 2, 3, 1, 0, 2, 3]) → true	true	OK	
canBalance([1, 2, 3, 1, 0, 1, 3]) → false	false	OK	
canBalance([1]) → false	false	OK	
canBalance([1, 1, 1, 2, 1]) → true	true	OK	
other tests
OK	

All Correct
 */
}
