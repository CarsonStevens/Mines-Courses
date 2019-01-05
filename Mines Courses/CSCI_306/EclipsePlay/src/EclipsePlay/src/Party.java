package EclipsePlay.src;

import java.util.ArrayList;

public class Party {
	private ArrayList<Friend> friends;

	public Party() {
		friends = new ArrayList<Friend>();
	}

	public void addPeople() {
 friends.add(new Friend("Mark", "baldwin"));
 friends.add(new Friend("Jane", "jsmith"));
 friends.add(new Friend("Joe", "jdoe"));
	}

	public void showPeople() {
		System.out.println("Who's Coming to the Party?");
		for (Friend person : friends) {
			System.out.println(person);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Party party = new Party();
		party.addPeople();
		party.showPeople();

	}

}
