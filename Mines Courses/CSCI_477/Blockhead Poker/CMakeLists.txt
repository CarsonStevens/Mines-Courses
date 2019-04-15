cmake_minimum_required(VERSION 3.13)
project(Blockhead_Poker)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Blockhead_Poker
        AlphaPlayer.cpp
        AlphaPlayer.h
        Bet.cpp
        Bet.h
        BetaPlayer.cpp
        BetaPlayer.h
        BetHistory.cpp
        BetHistory.h
        blockhead_poker.cpp
        Card.cpp
        Card.h
        Game.cpp
        Game.h
        Hand.cpp
        Hand.h
        HumanPlayer.cpp
        HumanPlayer.h
        Notes.txt
        Player.cpp
        Player.h)
