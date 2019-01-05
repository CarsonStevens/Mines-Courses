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

int main(){
    
    srand(time(0));
    RenderWindow window(VideoMode(800, 600), "Bubbles");
    
    vector<Bubbles> bubbles;
    int x;
    int y;
    
    for (int i = 0; i < 5; i++){
        Bubbles bubble;
        bubbles.push_back(bubble);
        //Tests to make sure that each bubble is getting generated with different values.
        //cout << "Velocity x: " << bubbles.at(i).getVelocityX() << endl;
    }
    
    
    
    while(window.isOpen()){

        Event event;
        while(window.pollEvent(event)){
            //Allows the user to close the window.
            if(event.type == Event::Closed){
                window.close();
            }
            
            //Adds a bubble when the Mouse button is released
            else if (event.type == Event::MouseButtonReleased){
                x = Mouse::getPosition(window).x;
                y = Mouse::getPosition(window).y;
                Bubbles bubble(x, y);
                bubbles.push_back(bubble);
            }
            
            //Deletes the most recent bubble when D is pressed
            else if (event.type == Event::KeyPressed){
                switch(event.key.code){
                        case (Keyboard::D):
                            bubbles.pop_back();
                            break;
                            
                        default:
                            break;
                }
            }
        }
        
        
        //Checks collisions for all the members of the bubbles vector.
        for(int i = 0; i < bubbles.size(); ++i){
            //Check for intersection on the x sides
            if((bubbles.at(i).getPosition().left > 800 - 2*bubbles.at(i).getRadius()) || (bubbles.at(i).getPosition().left < 0)){
                bubbles.at(i).hitSides();
            }
            //Checks for intersection on the y sides
            if((bubbles.at(i).getPosition().top > 600 - 2*bubbles.at(i).getRadius()) || (bubbles.at(i).getPosition().top < 0)){
                bubbles.at(i).hitTopBottom();
            }
            //Checks if the balls hit each other.
            for(int j = i+1; j < bubbles.size(); j++){
                if(bubbles.at(i).getPosition().intersects(bubbles.at(j).getPosition())){
                    bubbles.at(j).hitBubble();
                }
            }
            //Updates every balls position.
            bubbles.at(i).update();
        }
        
        window.clear(Color::Black);
        //cout << bubbles.size();
        
        //Draws all the bubbles.
        for(int i = 0; i < bubbles.size(); ++i){
            window.draw(bubbles.at(i).getShape());
        }
        
        window.display();
    }
    

}