import java.util.ArrayList;

public class Party {
	private ArrayList<Person> people;

	public Party() {
		people = new ArrayList<Person>();
	}

	public void addPeople() {
 people.add(new Person("Mark", "baldwin"));
 people.add(new Person("Jane", "jsmith"));
 people.add(new Person("Joe", "jdoe"));
	}

	public void showPeople() {
		System.out.println("Who's Coming to the Party?");
		for (Person person : people) {
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
