/* CSCI261 Section E Lab9A
 *
 * Description: Bubbles
 *
 * Author: Carson Stevens & Stephanie Holzschuh
 *
 */
 
#include <iostream>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

#include "bubble.h"

//Gives the bubble an intial condition
Bubble::Bubble(){
    _position.x = 400.0;
    _position.y = 300.0;
    _yVelocity = 0.4f;
    _xVelocity = 0.5f;
    _bubble.setFillColor(Color::White);
    _bubble.setPosition(_position);
    _bubble.setRadius(10);
}

//Returns the bubble
CircleShape Bubble::getShape(){
    return _bubble;
}

//Sets the x and y position from user input ***NEVER USED IN THIS PROGRAM        
void Bubble::setXPos(double x){
    if((x <= 600) && (x >= 0)){
        _position.x = x;
    }
    return;
}
void Bubble::setYPos(double y){
    if((y <= 800) && (7 >= 0)){
        _position.y = y;
    }
    return;
}

//Returns the position of the bubble
FloatRect Bubble::getPosition(){
    return _bubble.getGlobalBounds();
}

//Moves the bubble the amount of pixels compared to the velocity it is given
void Bubble::update(){
    _position.y += _yVelocity;
    _position.x += _xVelocity;
    
    _bubble.setPosition(_position);
}

//Tells the bubble what to do if it hits the top or bottom
void Bubble::hitTopBottom(){
    _yVelocity = -(_yVelocity);
}

//Tells the bubble what to do if it hits the sides.
void Bubble::hitSides(){
    _xVelocity= -(_xVelocity);
}
