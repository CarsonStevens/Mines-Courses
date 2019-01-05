import java.util.Arrays;

public class JavaBatPair {
	
	public String plusOut(String str, String word){
		String answer = "";
		int index = 0;
		
		while(index < str.length()) {
			/* startsWith check the beginning of the word or at an index.
			* The method uses the index to see if the word starts at the index.
			* If it does, then add it to the result and increment the current index by the length of the word just added.
			*/
			if(str.startsWith(word, index) == true){
				// DEBUG
				// System.out.println("if loop for word");
				answer += word;
				index += word.length();
			}
			else {
				// DEBUG
				// System.out.println("Plus loop");
				answer += "+";
				index++;
			}
		}
		return answer;		
	}
	
	public int[] fix34(int[] nums){
		int [] result = new int[nums.length];
		int size = 0;
		for(int i = 0; i < nums.length; i++) {
			if(nums[i] == 3) {
				size++;
			}
		}
		int [] others = new int[nums.length -(size * 2)];
		int [] index3s = new int[size];
		int index = 0;
		int counter = 0;
		for(int i = 0; i < nums.length; i++) {
			if(nums[i] == 3) {
				index3s[index] = i;
				index++;
				continue;
			}
			if(nums[i] != 3 & nums [i] != 4) {
				others[counter] = nums[i];
				counter++;
			}
		}
		for(int i = 0; i <= index3s.length-1; i++) {
			result[index3s[i]] = 3;
			result[index3s[i]+1] = 4;
		}
		int counter2 = 0;
		for(int i = 0; i < result.length; i++) {
			if(result[i] == 0) {
				result[i] = others[counter2];
				counter2++;
			}
		}
		return result;
	}
	
	public static void main(String[] args) {
//		int [] arr = {1, 3, 1, 4};
//		System.out.println(Arrays.toString(fix34(arr)));
	}

}
