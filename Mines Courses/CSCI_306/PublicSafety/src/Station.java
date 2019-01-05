import java.util.Scanner;

public class Station {
	static int badgeNum;
	private static final int MAX_DETECTIVES = 20;
	private static int currentAmount = 1;
	private String stationName;
	
	public void setStationName(String stationName) {
		this.stationName = stationName;
	}


	private Detective [] detectives= new Detective[MAX_DETECTIVES];
	
	
	public Station(String stationName) {
		super();
		this.stationName = stationName;
	}


	static void doHire(Boolean type) {
		Scanner scan = new Scanner(System.in);
		
		if(type == true) {
			// Hire city Detective
			System.out.print("New hire for " + this.stationCity + "...Enter detective's name:\t");
			String name = scan.next();
			++currentAmount;
		}
		else {
			// Hire University Detective
			System.out.print("New hire for " + this.stationUniversity + "...Enter detective's name:\t");
			String name = scan.next();
			++currentAmount;
		}
	}
}
