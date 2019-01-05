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

#pragma once

class Bubbles{
    public:
        Bubbles();
        Bubbles(double x, double y);
        void update();
        FloatRect getPosition();
        void setXPos(double x);
        void setYPos(double y);
        double getVelocityX();
        CircleShape getShape();
        void hitTopBottom();
        void hitSides();
        void hitBubble();
        double getRadius();
    private:
        CircleShape _bubble;
        Vector2f _position;
        double _yVelocity;
        double _xVelocity;
        double _radius;
        
    
};