package EclipsePlay.src;

/**
 * @author Mark Baldwin
 * @Editor Carson Stevens
 */
public class Friend {
	
	//Private Variables
	private String name;
	private String email;
	
	//Constructor
	public Friend(String name, String email) {
		super();
		this.name = name;
		this.email = email;
	}
	
	// Returns the Friend's email
	public String getEmail() {
		return email;
	}
	
	//To Set the friends email
	public void setEmail(String email) {
		this.email = email;
	}

	//Returns the friend's name
	public String getName() {
		return name;
	}

//	public String toString() {
//		return "\tName:\t" + getName() + "\n\tEmail:\t" + getEmail()+ "\n";
//	}
}
