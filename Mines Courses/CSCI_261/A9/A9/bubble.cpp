/* CSCI261 Section E Assignment 9
 *
 * Description: Bubbles
 *
 * Author: Carson Stevens
 *
 */
 
#include <iostream>
#include <ctime>
#include <vector>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;
#include "bubble.h"

Bubbles::Bubbles(){
    //Gives the bubbles random initial conditions.
    _position.x = rand() % 301 + 100;
    _position.y = rand() % 301 + 100;
    _yVelocity = (0.1f)*10*(((rand() % 26)/10.0));
    _xVelocity = (0.1f)*10*(((rand() % 26)/10.0));
    _radius = rand() % 41 + 10;
    _bubble.setFillColor(Color((rand() % 256), (rand() % 256), rand() % 256));
    _bubble.setPosition(_position);
    _bubble.setRadius(_radius);
}

Bubbles::Bubbles(double x, double y){
    //Gives the bubbles random conditions, but sets where the mouse was clicked to the initial position
    _position.x = x;
    _position.y = y;
    _yVelocity = (0.1f)*10*(((rand() % 26)/10.0));
    _xVelocity = (0.1f)*10*(((rand() % 26)/10.0));
    _radius = rand() % 41 + 10;
    _bubble.setFillColor(Color((rand() % 256), (rand() % 256), rand() % 256));
    _bubble.setPosition(_position);
    _bubble.setRadius(_radius);
}

//Gets the x velocity of a bubble
double Bubbles::getVelocityX(){
    return _xVelocity;
}

//Gets the y velocity of a bubble
double Bubbles::getRadius(){
    return _radius;
}

//Gets the bubble
CircleShape Bubbles::getShape(){
    return _bubble;
}

//Could be used to set the position of a bubble by user input ***NEVER USED IN CURRENT VERSION        
void Bubbles::setXPos(double x){
    if((x <= 600) && (x >= 0)){
        _position.x = x;
    }
    return;
}
void Bubbles::setYPos(double y){
    if((y <= 800) && (7 >= 0)){
        _position.y = y;
    }
    return;
}

//Gets the position of the bubble
FloatRect Bubbles::getPosition(){
    return _bubble.getGlobalBounds();
}

//Moves the bubble the amount of pixels compared to the velocity it is given
void Bubbles::update(){
    _position.y += _yVelocity;
    _position.x += _xVelocity;
    
    _bubble.setPosition(_position);
    return;
}

//Says what direction to change if the top of bottom is hit.
void Bubbles::hitTopBottom(){
    _yVelocity = -(_yVelocity);
    return;
}

//Says what direction to change if the sides are hit.
void Bubbles::hitSides(){
    _xVelocity = -(_xVelocity);
    return;
}

//Tells bubbles what to due if the bubble hits another bubble.
void Bubbles::hitBubble(){
    _xVelocity = -(_xVelocity);
    _yVelocity = -(_yVelocity);
    return;
}