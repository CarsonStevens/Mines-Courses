/*
 * Author: Carson Stevens
 * CSCI403 Project 6: Convert
 * Description: https://cs-courses.mines.edu/csci403/spring2019/projects/6-convert/convert.html
 * March 22, 2019
 */

/*
 * Drop tables from previous attempt or if they already exist
 */

--Step 1
DROP TABLE IF EXISTS artist CASCADE;
DROP TABLE IF EXISTS album CASCADE;
DROP TABLE IF EXISTS label CASCADE;

--Step 2
DROP TABLE IF EXISTS track CASCADE;

--Step 5
DROP TABLE IF EXISTS xref_member_of_group CASCADE;

--Step 6
DROP TABLE IF EXISTS genre CASCADE;

/*
 * PART 1: Constructing your database schema
 * Algorithm:
 *	 Step 1: Create the regular entities
 * 			-PK : id : serial
 *			-datatypes from project6 table
 */ 

--Artist info
CREATE TABLE artist(
	id serial,
	name TEXT NOT NULL,
	type TEXT NOT NULL,
	PRIMARY KEY(id)
);

--Album Info
CREATE TABLE album(
    	id serial,
    	title TEXT NOT NULL,
    	year numeric(4,0) NOT NULL,
    	PRIMARY KEY(id)
);

--Label info
CREATE TABLE label(
	id serial,
	name TEXT NOT NULL,
	location TEXT,
	PRIMARY KEY(id)
);

/* 
 * Step 2: Create the weak entities
 * 		-Attributes to columns
 * 		-Make FK for owning entity
 * 		-Make PK from borrowed and partial key
 */

-- Track info
CREATE TABLE track(
	name text,
	number TEXT NOT NULL,
	album_id INT REFERENCES album(id),
	PRIMARY KEY(name, album_id)
);

/*
 * Step 3: 1-1 relationships
 */
-- No 1-1 relationships

/*
 * Step 4: 1-to-Many relationships
 * 	-Using INT for FK and not serial **In Project Details**
 * Album: released by artist
 * Album: published by label
 */

ALTER TABLE album ADD COLUMN artist_id INT, ADD COLUMN label_id INT;
ALTER TABLE Album 
	ADD CONSTRAINT album_artist_id_fk
	FOREIGN KEY (artist_id)
	REFERENCES artist(id),
	ADD CONSTRAINT album_label_id_fk
	FOREIGN KEY (label_id)
	REFERENCES label(id);

/*
 * Step 5: Many-to-Many Relationships
 * 		-Create cross reference table
 *		-datatypes from project6 table
 * 			-Assign simple attributes (names and years)
 * 		-PK comprised of FKs (INTS instead of serial)
 * Individual: is member of group
 */
CREATE TABLE xref_member_of_group(
	member_name TEXT,
    	member_id INT REFERENCES artist(id),
    	group_name TEXT,
    	group_id INT REFERENCES artist(id),
    	begin_year numeric(4, 0),
    	end_year numeric(4, 0),
 	PRIMARY KEY(member_id, group_id)
);

/*
 * Step 6: Mulivalued Attributes
 * 		-Create Table
 * 			-Simple attributes to columns
 * 				-title
 *			-Attribute from owning entity
 *				-album_id
 * 			-PK from owning entity
 *				-album: album_id
 */

CREATE TABLE genre(
   	title TEXT,
    	album_id INT REFERENCES album(id),
    	PRIMARY KEY(title, album_id)
);



/*
 * PART 2: Populating your database
 *	**Use SELECT DISTINCT (from Project Details)**
 */ 

-- Insert data into artist table
INSERT INTO Artist(name, type) (
	SELECT DISTINCT
		artist_name,
		artist_type 
	FROM public.project6 
	UNION
	SELECT DISTINCT
		member_name,
		'Person'
	FROM public.project6
	WHERE artist_type='Group'
);

-- Insert/Join data into xref_member_of_group
INSERT INTO xref_member_of_group (
	SELECT DISTINCT
		project6.member_name,
		a2.id AS member_id,
		project6.artist_name AS group_name,
		a1.id AS group_id,
		project6.member_begin_year,
		project6.member_end_year 
	FROM project6 
	INNER JOIN artist a1 
		ON project6.artist_name=a1.name 
			AND project6.artist_type='Group' 
	INNER JOIN artist a2 
		ON project6.member_name=a2.name
);

-- Insert data into label table
INSERT INTO label(name, location) (
	SELECT DISTINCT
		label,
		headquarters 
	FROM project6
);

-- Insert data into album table
INSERT INTO album(title, year, artist_id, label_id) (
	SELECT DISTINCT
		project6.album_title,
		project6.album_year,
		artist.id,
		label.id 
	FROM project6
	INNER JOIN artist
		ON project6.artist_name = artist.name
	INNER JOIN label
		ON project6.label = label.name
);	

-- Insert data into genre table
INSERT INTO genre (
	SELECT DISTINCT
		project6.genre,
		album.id
	FROM project6
	INNER JOIN album
		ON project6.album_title = album.title
);

-- track
INSERT INTO track (
	SELECT DISTINCT
		project6.track_name,
		project6.track_number,
		album.id
	FROM project6
	INNER JOIN album 
		ON project6.album_title = album.title
);





/*
 * PART 3: Querying your database
 */ 

/*
 * 1. Get all members of The Who and their begin/end years 
 * 	with the group ordered by their starting year and name.
 * PROGRESS: WORKS
 */

SELECT member_name, begin_year, end_year 
	FROM xref_member_of_group 
	WHERE group_name='The Who' 
	ORDER BY begin_year, member_name;


/*
 * 2. Get all groups that Chris Thile has been a part of:
 * PROGRESS: WORKS
 */

SELECT name 
	FROM artist 
	INNER JOIN xref_member_of_group 
		ON xref_member_of_group.member_name='Chris Thile'
			AND xref_member_of_group.group_id=artist.id;


/*
 * 3. Get the album titles, years, and labels for all the 
 * 	albums put out by the band Talking Heads, ordered by year
 *
 * PROGRESS: WORKS
 */

SELECT album.title, album.year, label.name 
	FROM album
	INNER JOIN artist
		ON album.artist_id=artist.id 
			AND artist.name='Talking Heads' 
	INNER JOIN label
		ON album.label_id=label.id 
	ORDER BY album.year;


/*
 * 4. Get all albums (album, year, artist, and label) 
 * 	that Chris Thile has performed on, ordered by year
 *
 * PROGRESS: WORKS
 */

SELECT album.title, album.year, artist.name, label.name
	FROM album
	INNER JOIN artist
		ON album.artist_id=artist.id
			AND artist.name
	IN 
		(SELECT DISTINCT group_name 
			FROM xref_member_of_group 
			WHERE member_name='Chris Thile'
		UNION
			SELECT name 
			FROM artist 
			WHERE name='Chris Thile')
	INNER JOIN label
		ON album.label_id=label.id
	ORDER BY year;


/*
 * 5. Get all albums (artist, album, year) in the 'electronica'
 * 	genre ordered by year:
 *
 * PROGRESS: WORKS
 */

SELECT artist.name, album.title, album.year 
	FROM album
	INNER JOIN artist
		ON album.artist_id=artist.id
	INNER JOIN genre
		ON album.id=genre.album_id
			AND genre.title='electronica'
	ORDER BY album.year;


/*
 * 6. Get all the tracks on Led Zeppelin's Houses of the Holy 
 * 	in order by track number:
 * 
 * PROGRESS: WORKS
 */ 

SELECT track.name, track.number 
	FROM track 
	INNER JOIN album 
		ON track.album_id=album.id 
			AND album.title='Houses of the Holy' 
	ORDER BY track.number;

/*
 * 7. Get all genres that James Taylor has performed in:
 *
 * PROGRESS: WORKS
 */

SELECT DISTINCT genre.title 
	FROM genre
	INNER JOIN album 
		ON genre.album_id=album.id
	INNER JOIN artist 
		ON album.artist_id=artist.id 
			AND artist.name='James Taylor'; 


/*
 * 8. Get all albums published by a label headquartered in Hollywood:
 *
 * PROGRESS: WORKS
 */

SELECT artist.name, album.title, album.year, label.name 
	FROM album
	INNER JOIN artist
		ON album.artist_id=artist.id 
	INNER JOIN label 
		ON album.label_id=label.id
		AND location='Hollywood';

