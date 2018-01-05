# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 01:57:54 2018

@author: ndoannguyen
"""
import random
import numpy as np
from multiprocessing import Pool

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
        return self.__height
    
    def getWidth(self):
        """
            Exercise 1
            Get width of self
        """
        return self.__width
    
    def getBoardStatus(self):
        """
            Exercise 1
            Return the status of all the cells (a list of list of int)
        """
        return self.__cells
    
    def setBoardStatus(self, cells):
        """
            Exercise 1
            Override the current status of the board by value of cell
        """
        self.__cells = cells
    
    def checkCellValidity(self, cell):
        """
            Exercise 2
            Check if coordinates of the cell is valid
        """
        if cell[0] >= self.__height or cell[1] >= self.__width:
            raise BoardError("Cell " + str(cell) + "does not exist.")

    def getCellStatus(self, cell):
        """
            Exercise 2
            Check status of the cell
        """
        self.checkCellValidity(cell)
        return self.__cells[cell[0]][cell[1]]
    
    def isEmptyCell(self, cell):
        """
            Exercise 2
            Check if the cell is empty
        """
        return self.getCellStatus(cell) == Board.EMPTY
    
    def mark(self, value, cell):
        """
            Exercise 2
            Mark value to cell
        """
        self.checkCellValidity(cell)
        self.__cells[cell[0]][cell[1]] = value
    
    def getEmptyCells(self):
        """
            Exercise 2
            Return coordinates of all empty_cells
        """
        res = []
        for i in range(self.__height):
            for j in range(self.__width):
                if self.isEmptyCell((i, j)):
                    res.append((i, j))
        return res

    def getNeighbors(self, cell):
        """
            Exercise 3
            Return neighbors of a cell
        """
        neighbors = []
        x, y = cell[0], cell[1]
        if x >= 1:
            neighbors.append((x - 1, y))
        if y >= 1:
            neighbors.append((x, y - 1))
            if x >= 1:
                neighbors.append((x - 1, y - 1))
            if  x < self.__height - 1:
                neighbors.append((x + 1, y - 1))
        if x < self.__height - 1:
            neighbors.append((x + 1, y))
        if y < self.__width - 1:
            neighbors.append((x, y + 1))
            if x >= 1:
                neighbors.append((x - 1, y + 1))
            if x < self.__height - 1:
                neighbors.append((x + 1, y + 1))
        return neighbors
    
    def getARandomEmptyCell(self):
        """
            Exercise 9
            Get a random empty cell
        """
        empty_cells = self.getEmptyCells()
        # randint: endpoints included
        rand_int = random.randint(0, len(empty_cells) - 1)
        return empty_cells[rand_int]
        
    def getPotentialCells(self):
        """
            Complement of ex 12
            Return coordinates of all empty_cells whose neighbors is not empty
        """
        empty_cells = self.getEmptyCells()
        potential_cells = []
        for cell in empty_cells:
            neighbors = self.getNeighbors(cell)
            for neig in neighbors:
                if self.getCellStatus(neig) !=  Board.EMPTY:
                    potential_cells.append(cell)
                    break
        return potential_cells
        
    def getARandomPotentialCell(self):
        """
            Complement of ex 12
            Get a random potential cell
        """
        p_cells = self.getPotentialCells()
        if p_cells == []:
            p_cells = self.getARandomEmptyCell()
        # randint: endpoints included
        rand_int = random.randint(0, len(p_cells) - 1)
        return p_cells[rand_int]       


class Game:
    
    ACTIVE = 1
    INACTIVE = 0
    NOWINNER = 0
    X = 1
    O = 2
    HUMAN = 1
    MACHINE = 2 
    NB_AUTOMATIC_GAMES = 100
    
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
        return self.__board
    
    def getTurn(self):
        """
            Exercise 4
            Get whose's turn next
        """
        return self.__turn
    
    def isActive(self):
        """
            Exercise 4
            Get if the game is still active
        """
        return self.__status == Game.ACTIVE
    
    def getWinner(self):
        """
            Exercise 4
            Get winner
        """
        return self.__winner
        
    def getWinNumber(self):
        """
            Get the winNumber
        """
        return self.__winNumber
    
    def deactivate(self):
        """
            Exercise 4
            If the game is active, deactivate it
        """
        if self.__status == Game.INACTIVE:
            raise GameError("The game is already inactive.")
        self.__status = Game.INACTIVE
    
    def activate(self):
        """
            Exercise 4
            If the game is inactive, activate it
        """
        if self.__status == Game.ACTIVE:
            raise GameError("The game is already active.")
        self.__status = Game.ACTIVE
    
        
    def switchTurn(self):
        """
            Exercise 4
            Change the turn to the other player
        """
        self.__turn = Game.X + Game.O - self.__turn
    
    def declareWinner(self, player):
        """
            Exercise 4
            Declare a winner
        """
        if self.__winner != Game.NOWINNER:
            raise GameError("Winner already declared!")
        self.__winner = player
    
    def checkRight(self, player, cell):
        """
            Check if the player has right to mark.
            Right to mark = the game is still active, it's his turn and the cell is not empty
        """
        if not self.isActive():
            raise GameError("Game has finished")
        if self.__turn != player:
            raise GameError("It's not " + str(player) + "'s turn.")
        if not self.getBoard().isEmptyCell(cell):
            raise GameError("Cell " + str(cell) + " is not empty.")
        
    def mark(self, player, cell):
        """
            Exercise 5 and 8
            A player mark at cell
        """            
        self.checkRight(player, cell)
        self.getBoard().mark(player, cell)
        
        
        if self.isVictoryCell(cell):
            self.declareWinner(player)
            self.deactivate()
            return
        
        if self.isFull():
            self.deactivate()
            return
            
        self.switchTurn()
    
    def isVictoryCell(self, cell):
        """
            Exercise 5
            Check if after someone marks at cell, he becomes the winner
        """
        if self.getBoard().isEmptyCell(cell):
            return False
        return self.isVictoryCellByRow(cell) or self.isVictoryCellByColumn(cell) or self.isVictoryCellByFirstDiagonal(cell) or self.isVictoryCellBySecondDiagonal(cell)
    
    def isVictoryCellByRow(self, cell):
        """
            Check if after someone marks at cell, he becomes the winner because x x x x x on a row
        """
        player = self.getBoard().getCellStatus(cell)
        x = cell[0]
        y = cell[1]
        win = False
        for i in range( max(y - self.__winNumber + 1, 0),  min(y, self.getBoard().getWidth() - self.__winNumber) + 1):
            current_check = True
            for j in range(0, self.__winNumber):
                if self.getBoard().getCellStatus((x, i + j)) != player:
                    current_check = False
                    break
            if current_check == True:
                win = True
                break
        return win
    
    def isVictoryCellByColumn(self, cell):
        """
            Check if after someone marks at cell, he becomes the winner because x x x x x on a column
        """
        player = self.getBoard().getCellStatus(cell)
        x = cell[0]
        y = cell[1]
        win = False
        for i in range( max(x - self.__winNumber + 1, 0),  min(x, self.getBoard().getHeight() - self.__winNumber) + 1):
            current_check = True
            for j in range(0, self.__winNumber):
                if self.getBoard().getCellStatus((i + j, y)) != player:
                    current_check = False
                    break
            if current_check == True:
                win = True
                break
        return win
    
    def isVictoryCellByFirstDiagonal(self, cell):
        """
            Check if after someone marks at cell, he becomes the winner because x x x x x on a NorthWest - SouthEast diagonal
        """
        player = self.getBoard().getCellStatus(cell)
        x = cell[0]
        y = cell[1]
        win = False
        
        for i in range(0, self.__winNumber):
            if x - i < 0 or y - i < 0 or x - i + self.__winNumber > self.getBoard().getHeight() or y - i + self.__winNumber > self.getBoard().getWidth():
                continue
            current_check = True
            for j in range(0, self.__winNumber):
                if self.getBoard().getCellStatus((x - i + j, y - i + j)) != player:
                    current_check = False
                    break
            if current_check == True:
                win = True
                break
        return win
    
    def isVictoryCellBySecondDiagonal(self, cell):
        """
            Check if after someone marks at cell, he becomes the winner because x x x x x on a NorthEast - SouthWest diagonal
        """
        player = self.getBoard().getCellStatus(cell)
        x = cell[0]
        y = cell[1]
        win = False
        
        for i in range(0, self.__winNumber):
            if x - i < 0 or y + i > self.getBoard().getWidth() - 1 or x - i + self.__winNumber > self.getBoard().getHeight() or y + i - self.__winNumber < -1:
                 continue
            current_check = True
            for j in range(0, self.__winNumber):
                if self.getBoard().getCellStatus((x - i + j, y + i - j)) != player:
                    current_check = False
                    break
            if current_check == True:
                win = True
                break
        return win

    def isFull(self):
        """
            Exercise 7
            Check if the board is full
        """
        return self.getBoard().getEmptyCells() == []

    def evaluateLastStepRandomly(self, player):
        """
            Exercise 9
            Evaluate last step by generating automatically the following steps and check the winner
        """
        while self.isActive():
            next_cell = self.getBoard().getARandomEmptyCell()
            self.mark(self.getTurn(), next_cell)
            
        if self.getWinner() == player:
            return 1
        elif self.getWinner() == Game.NOWINNER:
            return 0
        else:
            return -1
    
    def evaluateLastStepPotentially(self, player):
        """
            Complement of ex 12
            Evaluate last step by generating automatically the following steps and check the winner
        """
        while self.isActive():
            next_cell = self.getBoard().getARandomPotentialCell()
            self.mark(self.getTurn(), next_cell)
            
        if self.getWinner() == player:
            return 1
        elif self.getWinner() == Game.NOWINNER:
            return 0
        else:
            return -1    
    
    def generateGameCopy(self):
        """
            Exercise 10
            Generate a game copy
        """
        game_copy = Game(self.getBoard().getHeight(), self.getBoard().getWidth(), self.getTurn(), self.getWinNumber())
        cells = self.getBoard().getBoardStatus()
        cells2 = []
        for i in range(self.getBoard().getHeight()):
            cells2.append([])
            for j in range(self.getBoard().getWidth()):
                cells2[i].append(cells[i][j])
        game_copy.getBoard().setBoardStatus(cells2)
        game_copy.__turn = self.getTurn()
        return game_copy
    
    def decideNextStep(game, player, algo):
        """
            Decide next step
            A static method taking a game, a player, an algo (1, 2, 3) and decide which cell should be chosen
        """
        return algo(game, player)
        #FINISHED. DO NOT TOUCH
        
    def algo1(self, player):
        """
            Exercise 11
            An algo to decide next step
        """
        empty_cells = self.getBoard().getEmptyCells()
        efficiency = [0] * len(empty_cells)
        for idx, cell in enumerate(empty_cells):
            for i in range(Game.NB_AUTOMATIC_GAMES):
                game_copy = self.generateGameCopy()
                game_copy.mark(self.getTurn(), cell)
                efficiency[idx] += game_copy.evaluateLastStepRandomly(player)
        return empty_cells[np.argmax(efficiency)]
    
    def algo2(self, player):
        """
            Exercise 12
            Another algo to make next step
        """
        p_cells = self.getBoard().getPotentialCells()
        efficiency = [0] * len(p_cells)
        for idx, cell in enumerate(p_cells):
            for i in range(Game.NB_AUTOMATIC_GAMES):
                game_copy = self.generateGameCopy()
                game_copy.mark(self.getTurn(), cell)
                efficiency[idx] += game_copy.evaluateLastStepPotentially(player)
        return p_cells[np.argmax(efficiency)]
        
    def algo3(self, player):
        """
            Exercise 13
            An algo to make next step
        """
        p_cells = self.getBoard().getPotentialCells()
        my_pools = Pool(8)
        efficiency = my_pools.map(generateAndEvaluate, [(self, p_cells[i], player) for i in range(len(p_cells))])
        return p_cells[np.argmax(efficiency)]

def generateAndEvaluate(argument):
    """
        
        Make public a function        
    """
    game, cell, player = argument[0], argument[1], argument[2]
    score = 0
    for i in range(Game.NB_AUTOMATIC_GAMES):
        game_copy = game.generateGameCopy()
        game_copy.mark(game.getTurn(), cell)
        score += game_copy.evaluateLastStepPotentially(player)
    return score	
