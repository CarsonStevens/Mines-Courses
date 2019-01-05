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


int main(){
    
    RenderWindow window(VideoMode(800, 600), "Bubbles");
    Bubble bubble;
    
    while(window.isOpen()){

        Event event;
        while( window.pollEvent( event) ){
            //Allows the user to close the window.
            if(event.type == Event::Closed){
                window.close();
            }
        }
        
        //Gives the dimensions of the window to tell the bubble when to rebound
        if((bubble.getPosition().left > 800 - 20) || (bubble.getPosition().left < 0)){
            bubble.hitSides();
        }
        
        //Gives the dimensions of the window to tell the bubble when to rebound
        if((bubble.getPosition().top > 600 - 20) || (bubble.getPosition().top < 0)){
            bubble.hitTopBottom();
        }
        
        //Updates the bubble position
        bubble.update();
        //Clears the last frame
        window.clear( Color::Black );
        //Draws the bubble in its new spot
        window.draw(bubble.getShape());
        //Displays the window.
        window.display();
    }
    
    return 0;
}