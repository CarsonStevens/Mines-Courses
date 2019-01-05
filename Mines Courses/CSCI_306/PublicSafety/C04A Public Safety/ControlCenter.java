/**
 * ControlCenter class
 * Used as part of the PublicSafety exercise, CSCI306
 * 
 * @author Mark Baldwin
 * @author Cyndi Rader
 * 
 */
import java.util.ArrayList;

/**
 *  Creates and test PublicSafety classes 
 *
 */
public class ControlCenter {
	// You'll learn about ArrayList soon
	private ArrayList<PublicSafety> psOffices;
	
	/**
	 * Constructor, sets up array list of public safety offices
	 */
	public ControlCenter() {
		psOffices = new ArrayList<PublicSafety>();
	}
	


	/**
	 * Create offices to test.
	 * 
	 * In a real system we would prompt for office names. To simplify
	 * grading and testing, we will hard code names and the sequence 
	 * of hires in the following methods
	 */
	public void createOffices() {
		psOffices.add(new PublicSafety("CSM", "Golden"));
		psOffices.add(new PublicSafety("CU", "Boulder"));
	}
	
	/**
	 * Hire from several stations to show the effect on badge number
	 */
	public void doHiring() {

		PublicSafety psOffice1 = psOffices.get(0);
		PublicSafety psOffice2 = psOffices.get(1);
		/*
		 *  Using an enumerated type would be better - we'll learn that soon 
		 *  For now, the PublicSafety class should have a doHire method. 
		 *  The parameter to doHire is a boolean. 
		 *  - If true, tell the city station to hire one detective. 
		 *  - If false, tell the university station to hire one detective.
		 */
		
		psOffice1.doHire(true);
		psOffice1.doHire(false);
		psOffice1.doHire(true);
		psOffice2.doHire(false);
		psOffice2.doHire(true);
		psOffice1.doHire(true);
		
		// Now we do a loop to show that stations ensure they don't hire more
		// than the max # of detectives. See figure 1 in assignment writeup.
		for (int i=4; i<=6; i++) {
			psOffice1.doHire(true);
		}
	}
	
	/**
	 * Print the results of test
	 */
	public void printAllDetectives() {
		System.out.println("\nPrinting All Detectives");
		for (PublicSafety office : psOffices) {
			office.printDetectiveLists();
		}
	}

	/** 
	 * Main driver for testing
	 * @param args Input arguments for main (unused)
	 */
	public static void main(String[] args) {
		ControlCenter cCenter = new ControlCenter(); 
		cCenter.createOffices();
		cCenter.doHiring();
		cCenter.printAllDetectives();
	}

}
