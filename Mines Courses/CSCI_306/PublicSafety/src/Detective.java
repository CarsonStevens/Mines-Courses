
public class Detective {
	private String officerName;
	private int officerBadgeNum;
	
	public Detective(String officerName, int officerBadgeNum) {
		super();
		this.officerName = officerName;
		this.officerBadgeNum = officerBadgeNum;
	}

	public String getName() {
		return officerName;
	}

	@Override
	public String toString() {
		return "Detective [officerName=" + officerName "]";
	}
	
	
	
}
