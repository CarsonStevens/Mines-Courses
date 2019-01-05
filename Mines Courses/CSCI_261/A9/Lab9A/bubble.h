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

class Bubble{
    public:
        Bubble();
        void update();
        FloatRect getPosition();
        void setXPos(double x);
        void setYPos(double y);
        CircleShape getShape();
        void hitTopBottom();
        void hitSides();
    private:
        CircleShape _bubble;
        Vector2f _position;
        double _yVelocity;
        double _xVelocity;
    
};