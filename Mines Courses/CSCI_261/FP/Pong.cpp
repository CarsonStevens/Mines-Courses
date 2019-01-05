/* CSI261 Final Project
 *
 * Description: To create a game of Pong with a twist (See final.txt)
 * Section E
 * Author: Carson Stevens
 *
 */
 
#include <iostream>
#include <ctime>
#include <ios>
#include <iomanip>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

#include "Pong.h"


//////////////////////////////////////////////////////////////////////////////////////
//BARRICADE CLASS FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////

//Parameterized constructor to establish different speeds for different levels.
Barricade::Barricade(double speed){
    if(speed < 1){
        _barricadeSpeed = speed;
    }
    else{
        _barricadeSpeed = speed;
    }
    _position.x = rand() % 400 + 100;
    _position.y = rand() % 620 + 100;
    _barricade.setPosition(_position);
    _barricade.setSize(Vector2f(5, 80));
}

//Used to speed up the barricade for a level
void Barricade::changeSpeed(double y){
    _barricadeSpeed = _barricadeSpeed * y;
    return;
}

//Causes the barricade to switch directions(used when it collides with an object)
void Barricade::barricadeCollision(){
    _barricadeSpeed = -(_barricadeSpeed);
    return;
}

//Used to trouble shoot and test. Gets the speed of the moving barricade
double Barricade::getSpeed(){
    return _barricadeSpeed;
}

//Changes the position relative to the speed of the barricade.
void Barricade::update(){
    _position.y += _barricadeSpeed;
    _barricade.setPosition(_position);
    return;
}

//Gets the GlobalBoundaries of the Barricades
FloatRect Barricade::getPosition(){
    return _barricade.getGlobalBounds();
}

//Returns the barricade. Useful for drawing the shape
RectangleShape Barricade::getShape(){
    return _barricade;
}



//////////////////////////////////////////////////////////////////////////////////////
//PADDLE CLASS FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////


//Parameterized Constructor for the Paddle
Paddle::Paddle(double x, double y, double speed){
    _position.x = x;
    _position.y = y;
    _paddleSpeed = speed;
    
    _paddle.setSize(Vector2f(5, 100));
    _paddle.setPosition(_position);
}

//Returns the GlobalBoundaries for the paddle
FloatRect Paddle::getPosition(){
    return _paddle.getGlobalBounds();
}

//Returns the paddles shape(useful for drawing the shape)
RectangleShape Paddle::getShape(){
    return _paddle;
}

//Moves the paddle up relative to the paddle speed
void Paddle::moveUp(){
    _position.y -= _paddleSpeed;
    return;
}

//Changes the positions by subtracting the value of the paddles speed
void Paddle::moveDown(){
    _position.y += _paddleSpeed;
    return;
}

//Updates the position of the paddle relative to the new position set in the
//functions above
void Paddle::update(){
    _paddle.setPosition(_position);
    return;
}


//////////////////////////////////////////////////////////////////////////////////////
//BALL CLASS FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////

//Parameterized constructor for the ball
Ball::Ball(double x, double y, double xV, double yV, double r){
    _position.x = x;
    _position.y = y;
    _xVelocity = xV;
    _yVelocity = yV;
    
    _ball.setRadius(r);
    _ball.setFillColor(Color::White);
    _ball.setPosition(_position);
}

//Used to change the speed of the ball (used in a certain level)
void Ball::changeSpeed(double x, double y){
    _xVelocity = _xVelocity * x;
    _yVelocity = _yVelocity * y;
    return;
}

//Used to change the color of the ball (used in a certain level)
void Ball::changeColor(){
    _ball.setFillColor(Color((rand() % 256), (rand() % 256), rand() % 256));
    return;
}

//Gets the GlobalBoundaries of the ball
FloatRect Ball::getPosition(){
    return _ball.getGlobalBounds();
}

//Gets the ball (Useful for drawing the ball shape)
CircleShape Ball::getShape(){
    return _ball;
}

//Gets the x velocity. Never used in the program but was used to test and trouble shoot
//uniquenesss of velocity
double Ball::getXVelocity(){
    return _xVelocity;
}

//Gets the y velocity. Never used in the program but was used to test and trouble shoot
//uniquenesss of velocity
double Ball::getYVelocity(){
    return _yVelocity;
}

//Used when the ball hits the top or bottom. Changes the y direction
void Ball::hitSides(){
    _yVelocity = -_yVelocity;
    return;
}

//Used when the ball hits a barricade. Changes the x direction
void Ball::hitBarricade(){
    _xVelocity = -(_xVelocity);
    return;
}

//Used when the ball hits the paddle. Changes the x directions
void Ball::hitPaddle(){
    _position.x += 10;
    _xVelocity = -(_xVelocity);
    return;
}

//Used when the ball hits the right side. Changes the x direction
void Ball::score(){
    _position.x -= 10;
    _xVelocity = -(_xVelocity);
    return;
}

//Used when the ball hits the right side. Changes the x direction
void Ball::lose(){
    _position.x += 10;
    _xVelocity = -(_xVelocity);
    return;
}

//Updates the position of the ball relative to what the velocities changed
//in the functions above.
void Ball::update(){
    _position.y += _yVelocity;
    _position.x += _xVelocity;
    
    _ball.setPosition(_position);
}

////////////////////////////////////////////////////////////////////////////////
/////Runs the game
////////////////////////////////////////////////////////////////////////////////

int runGame(){
        //Seeds the randomness for the program
    srand(time(0));
    
    //Initializes the window size
    int windowWidth = 700;
    int windowHeight = 800;
    RenderWindow window(VideoMode(windowWidth, windowHeight), "Pong");
    window.clear(Color::Black);
    
    //Defines the score and life variables
    int score = 0;
    int scoreDisplay;
    int lives = 3;
    int level = 1;
    int gameOver = 0;
    
    //Creates the parameterized paddle
    Paddle paddle (windowWidth/20, (windowHeight-100)/2, 10*0.2f);
    
    //Creates the parameterized ball
    Ball ball (windowWidth/2, (windowHeight-100)/2, 0.7f, 0.7f, 10);
    
    //Creates the vector that will hold the barricades that are place on the screen.
    vector<Barricade> barricades;
    
    //Creates the top of the game board
    RectangleShape top;
    top.setSize(Vector2f(windowWidth, 5));
    top.setPosition(Vector2f(0, 100));
    top.setFillColor(Color::White);
    
    
    //Defines the font that will be used throughout the game
    Font myFont;
    //Makes sure that the font loads correctly
    if( !myFont.loadFromFile( "Pacifico.ttf" ) ) {
	    cerr << "Error loading font" << endl;
    	    return -1;
	}
	
    
    
    
    while(window.isOpen()){
        
        //Lets the window be closed
        Event event;
        while(window.pollEvent(event)){
            switch(event.type){

                case (Event::Closed):
                    window.close();
                    break;
            
                // Add switch cases here for easier use in furture
                
                default:
                    break;
            }
        }
        
        //Handles Starting the game after the space bar is pressed
        if (Keyboard::isKeyPressed(Keyboard::Space)){
            gameOver = 1;
        }
        
        //Does the movement for the paddles in the down direction
        if(Keyboard::isKeyPressed(Keyboard::Down)){
            //The - 100 referes to the height of the paddle in the y direction
            if(paddle.getPosition().top < (windowHeight - 100)){
                paddle.moveDown();
            }
            
        }
        
        //Does the movement for the paddles in up direction
        if(Keyboard::isKeyPressed(Keyboard::Up)){
            //The 100 here defines the space used for the hud
            if(paddle.getPosition().top > 100){
                paddle.moveUp();
            }    
        }
        
        //Makes the ball rebound if it hits the top or the bottom. The 20 refers to the diameter of the ball.
        if(ball.getPosition().top < 120 || ball.getPosition().top + 20 > windowHeight){
            ball.hitSides();
        }
        
        //This says if the ball hits the right side of the screen, add to score and rebound ball
        if(ball.getPosition().left > windowWidth){
            ball.score();
            ++score;
            //bongSound.play();
            
            //When you score, adds a barricade; up to 10 barricades
            if(score < 11){
                Barricade barricade(0.0f);
                
                //TEST: Makes sure that barricades start with 0 speed
                //cout << barricade.getSpeed() << endl;
                
                barricades.push_back(barricade);
            }
            //Once 10 barricades are reached, it makes moving barricades.
            else if (score >= 11){
                Barricade barricade((0.1f)*10*(((rand() % 8)/10.0)));
                
                //TEST: Makes sure that the barricades were getting random speeds that were different
                //cout << barricade.getSpeed() << endl;
                
                //Erases the first barricade that was created that is still on the board
                barricades.erase(barricades.begin());
                //Makes a new barricade to replace it
                barricades.push_back(barricade);
                
                if (score >= 30){
                    ball.changeSpeed(1.005, 1.005);
                }
                
                if (score >= 40){
                    for(int j = 0; j < barricades.size(); j++){
                        barricades.at(j).changeSpeed(1.01);
                    }
                }
            }
            
            //TEST: Prints to the consol to makes sure that the size of the barricade vector was changing and doing
            //      what I wanted.
            //cout << barricades.size() << endl;
        }
        if (score >= 20){
            ball.changeColor();
        }
        
        //Says if the ball hits the left side to subtract a life and rebound the ball.
        if(ball.getPosition().left < 0 ){
            ball.lose();
            lives--;
            
            //Causes the game to reach the gameover screen
            if(lives == 0){
                gameOver = -1;
            }
        }
        
        
        //Checks if the ball hits the paddle. If it does, changes the balls direction.
        if(ball.getPosition().intersects(paddle.getPosition())){
            ball.hitPaddle();
        }
        
        //Checks if the ball intersect any of the barricades on the field. If so, causes the ball to change direction.
        for(int i = 0; i < barricades.size(); i++){
            if(ball.getPosition().intersects(barricades.at(i).getPosition())){
                ball.hitBarricade();
                // bingSound.play();
            }
            
            //Tells the barricades when to switch directions
            if(barricades.at(i).getPosition().top < 100 || barricades.at(i).getPosition().top + 80 > windowHeight){
                barricades.at(i).barricadeCollision();
            }
        }
        
        //Does the intro screen
        if (gameOver == 0){
            string introText;
            introText = "\n\t\t  Welcome to Pong with a twist!\n   The goal is to get the ball to the right\n \tside of the screen at any cost. Each\n\t\tlevel will make the game harder.\n\t\t\t\t\tGOODLUCK!\n  Use the 'Up' and 'Down' arrows to move\n\t\t\t\t\t  your paddle\n\t\t  Press the Space Bar to start!";
            Text intro;
            intro.setFont(myFont);
            intro.setCharacterSize(40);
            intro.setString(introText);
            intro.setPosition(Vector2f(0,0));
            window.draw(intro);
            
        }
        //Updates and draws everything on the screen.
        else if (gameOver > 0){
            
            level = score/10 + 1;
            
            //Handles the live hud feed
            //Sets font variables
            Text hud;
            hud.setFont(myFont);
            hud.setCharacterSize(40);
            hud.setPosition(Vector2f(0, 30));
            stringstream ss;
            ss << "Score:\t" << score << "\tLives:\t" << lives << "\tLevel:\t" << level;
            hud.setString( ss.str().c_str() );
            
            //Updates the ball's position
            ball.update();
            
            //Updates the paddle's position
            paddle.update();
            
            //Updates the position of all the barricades
            for(int i = 0; i < barricades.size(); i++){
                barricades.at(i).update();
            }
            
            //Clears the window from the last frame.
            window.clear(Color::Black);
            
            //Draws everything
            window.draw(hud);
            window.draw(paddle.getShape());
            window.draw(top);
            for(int i = 0; i < barricades.size(); i++){
                window.draw(barricades.at(i).getShape());
            }
            window.draw(ball.getShape());
        }
        
        //Does the game over screen
        else {
            
            level = score/10 + 1;
            scoreDisplay = score;
            //Clears the last from
            window.clear(Color::Black);
            
            //Defines the game over text.
            Text done;
            done.setFont(myFont);
            done.setCharacterSize(80);
            done.setString("\tGAME OVER!\t\t");
            done.setStyle(Text::Bold | Text:: Underlined);
            done.setPosition(Vector2f(0, 450));
            
            //Defines the score text for the end screen
            Text score;
            score.setFont(myFont);
            score.setCharacterSize(40);
            string exitStats ="\t\t\tLevel: " + to_string(level) + " \t\t   Score: " + to_string(scoreDisplay);
            score.setString(exitStats);
            score.setStyle(Text::Bold);
            score.setPosition(Vector2f(0, 290));
            
            //Draws the game over text.
            window.draw(done);
            window.draw(score);
        }
        
        //Displays the frame that was just drawn.
        window.display();
    }
    
    return 0;
}