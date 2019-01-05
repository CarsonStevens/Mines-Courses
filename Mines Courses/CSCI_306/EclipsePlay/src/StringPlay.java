//Author: Carson Stevens
//CWID: 10758778
//Date: 8/27/2018

import java.awt.Point;

public class StringPlay {

	public static void main(String[] args) {
		// String Manipulation
		String cpp = "C++ is cool";
		String java = "I love Java";
		String combo = java.substring(7) + cpp.substring(3);
		System.out.println(combo);
		
		//Point Comparison
		Point pt1 = new Point();
		Point pt2 = new Point();
		System.out.println(pt1 == pt2); // Tests the objects address, not contents
		
		//String Comparison
		String compare1 = "JAVA";
		String compare2 = "java";
		System.out.println("\nComparing the strings " + compare1 + " and " + compare2);
		
		//Exploration of "equals()"
		System.out.println("Comparison with 'equals':\t" + compare1.equals(compare2));
		
		//Exploration of "equalsIgnoreCase()"
		System.out.println("Comparison with 'equalsIgnoreCase':\t" + compare1.equalsIgnoreCase(compare2));
		
	}

}
