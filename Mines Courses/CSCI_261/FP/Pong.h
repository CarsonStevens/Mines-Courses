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

#pragma once

class Barricade{
    public:
        Barricade(double speed);
        FloatRect getPosition();
        RectangleShape getShape();
        void update();
        void barricadeCollision();
        double getSpeed();
        void changeSpeed(double y);
        
    private:
        RectangleShape _barricade;
        Vector2f _position;
        double _barricadeSpeed;
};

class Paddle{
    public:
        Paddle(double x, double y, double speed);
        RectangleShape getShape();
        FloatRect getPosition();
        void moveUp();
        void moveDown();
        void update();
    
    private:
        Vector2f _position;
        RectangleShape _paddle;
        double _paddleSpeed;
};

class Ball{
    public:
        Ball(double x, double y, double xV, double yV, double r);
        FloatRect getPosition();
        CircleShape getShape();
        double getXVelocity();
        double getYVelocity();
        void hitSides();
        void hitPaddle();
        void hitBarricade();
        void score();
        void lose();
        void update();
        void changeColor();
        void changeSpeed(double x, double y);
        
    private:
        Vector2f _position;
        CircleShape _ball;
        double _xVelocity;
        double _yVelocity;
        
};

//The function that runs the game.
int runGame();