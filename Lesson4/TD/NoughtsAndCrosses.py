# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 01:57:54 2018

@author: ndoannguyen
"""

class BoardError(Exception):
    def __init__(self, value):
        self.__value = value
    def __str__(self):
        return repr(self.__value)

class GameError(Exception):
    def __init__(self, value):
        self.__value = value
    def __str__(self):
        return repr(self.__value)

class Board:
    
    EMPTY = 0    
    X = 1
    O = 2
    CHARS = [" ", "X", "O"]
    
    def __init__(self, height, width):
        """
            Create a board with defined height and width
            Fill 0 (empty) to every cell of the board
        """
        self.__height = height
        self.__width = width
        self.__cells = []
        for i in range(height):
            self.__cells.append([])
            for j in range(width):
                self.__cells[i].append(Board.EMPTY)
    
        
    def draw(self):
        """
            Sketch the board
        """
        s = "-"
        for j in range(self.__width):
            s += "--"
        s += "\n"
        for i in range(self.__height):
            s += "|"
            for j in range(self.__width):
                s += str(Board.CHARS[self.__cells[i][j]]) + "|"
            s += "\n-"
            for j in range(self.__width):
                s += "--"
            s += "\n"
        return s
        
    def getHeight(self):
        """
            Exercise 1
            Get height of self
        """
        #TODO
    
    def getWidth(self):
        """
            Exercise 1
            Get width of self
        """
        #TODO
    
    def getBoardStatus(self):
        """
            Exercise 1
            Return the status of all the cells (a list of list of int)
        """
        #TODO
    
    def setBoardStatus(self, cells):
        """
            Exercise 1
            Override the current status of the board by value of cell
        """
        #TODO

    def getCellStatus(self, cell):
        """
            Exercise 2
            Check status of the cell
        """
        #TODO
    
    def isEmptyCell(self, cell):
        """
            Exercise 2
            Check if the cell is empty
        """
        #TODO
    
    def mark(self, value, cell):
        """
            Exercise 2
            Mark value to cell
        """
        #TODO
    
    def getEmptyCells(self):
        """
            Exercise 2
            Return coordinates of all empty_cells
        """
        #TODO

    def getNeighbors(self, cell):
        """
            Exercise 3
            Return neighbors of a cell
        """
        #TODO
    
    def getARandomEmptyCell(self):
        """
            Exercise 9
            Get a random empty cell
        """
        #TODO


class Game:
    
    ACTIVE = 1
    INACTIVE = 0
    NOWINNER = 0
    X = 1
    O = 2
    HUMAN = 1
    MACHINE = 2 
    
    def __init__(self, height, width, firstTurn, winNumber):
        """
            Begin a game and define which attributes are necessary to describe the game status at some moment
        """
        self.__winNumber = winNumber
        self.__board = Board(height, width)
        self.__turn = firstTurn
        self.__status = Game.ACTIVE
        self.__winner = Game.NOWINNER
    
    def draw(self):
        """
            Draw the board and status
        """
        print self.getBoard().draw()
    
    def getBoard(self):
        """
            Exercise 4
            Get the board
        """
        #TODO
    
    def getTurn(self):
        """
            Exercise 4
            Get whose's turn next
        """
        #TODO
    
    def isActive(self):
        """
            Exercise 4
            Get if the game is still active
        """
        #TODO
    
    def getWinner(self):
        """
            Exercise 4
            Get winner
        """
        #TODO
        
    def getWinNumber(self):
        """
            Get the winNumber
        """
        #TODO
    
    def deactivate(self):
        """
            Exercise 4
            If the game is active, deactivate it
        """
        #TODO
    
    def activate(self):
        """
            Exercise 4
            If the game is inactive, activate it
        """
        #TODO
        
    def switchTurn(self):
        """
            Exercise 4
            Change the turn to the other player
        """
        #TODO
    
    def declareWinner(self, player):
        """
            Exercise 4
            Declare a winner
        """
        #TODO

    def mark(self, player, cell):
        """
            Exercise 5 and 8
            A player mark at cell
        """            
        #TODO
        
    def isVictoryCell(self, cell):
        """
            Exercise 6
            Check if after someone marks at cell, he becomes the winner
        """
        #TODO

    def isFull(self):
        """
            Exercise 7
            Check if the board is full
        """
        #TODO

    def evaluateLastStepRandomly(self, player):
        """
            Exercise 9
            Evaluate last step by generating automatically the following steps and check the winner
        """
        #TODO
    
    def generateGameCopy(self):
        """
            Exercise 10
            Generate a game copy
        """
        #TODO
    
    def decideNextStep(self, player, algo):
        """
            Decide next step

        """
        return algo(self, player)
        #FINISHED. DO NOT TOUCH
        
    def algo1(self, player):
        """
            Exercise 11
            An algo to decide next step
        """
        #TODO
    
    def algo2(self, player):
        """
            Exercise 12
            Another algo to make next step
        """
        #TODO
        
    def algo3(self, player):
        """
            Exercise 13
            An algo to make next step
        """
        #TODO