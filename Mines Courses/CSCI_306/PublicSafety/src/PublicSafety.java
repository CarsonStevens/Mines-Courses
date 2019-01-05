import java.util.Scanner; 

public class PublicSafety {
	
	private Station stationUniversity;
	private Station stationCity;
	
	
	public PublicSafety(String stationUniversity, String stationCity) {
		super();
		this.stationUniversity = setStationName(stationUniversity);
		this.stationCity = setStationName(stationCity);
	}
	


	public String printDetectiveLists() {
		System.out.println("Printing All Detectives");
		
	}
	
}
