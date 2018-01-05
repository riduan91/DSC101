# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 01:52:44 2018

@author: ndoannguyen
"""

from NoughtsAndCrosses_Solution import Board, Game
import sys, time

def test0():
    board = Board(5, 5)
    print board.draw()
    print "Test 0 OK"
    return

def test1():
    board = Board(5, 4)
    checker = 1
    if board.getHeight() != 5:
        print "getHeight() not OK."
        print "Expected: 5"
        print "Received: ", board.getHeight()
        checker = 0
    
    if board.getWidth() != 4:
        print "getWidth() not OK"
        print "Expected: 4"
        print "Received: ", board.getWidth()
        checker = 0
    
    cells = [[0,0,0,0], [0,1,0,0], [0,1,2,0], [0,2,0,0], [0,0,2,0]]
    board.setBoardStatus(cells)
    if board.getBoardStatus() != cells:
        print "setBoardStatus(cells) or getBoardStatus() not OK."
        print "Expected: ", cells
        print "Received: ", board.getBoardStatus()
        checker = 0
        
    print board.draw()
    
    if checker == 1:
        print "Test 1 OK"
    else:
        print "Test 1 not OK"

def test2():
    board = Board(5, 5)
    checker = 1
    cells = [[0,0,0,0,0], [0,1,0,0,0], [0,1,2,0,0], [0,2,0,0,0], [0,0,2,0,0]]
    board.setBoardStatus(cells)
    if board.getCellStatus((2, 2)) != 2:
        print "getCellStatus() not OK."
        print "Expected: ", cells
        print "Received: ", board.getBoardStatus()   
        checker = 0
    if not board.isEmptyCell((3, 2)):
        print "isBoardStatus() not OK."
        print "Expected: ", cells
        print "Received: ", board.getBoardStatus()
        checker = 0
    board.mark(1, (1, 2))
    if board.getCellStatus((1, 2)) != 1:
        print "mark() not OK."
        print "Expected: 1"
        print "Received: ", board.getCellStatus((1, 2))      
        checker = 0
    
    if set(board.getEmptyCells()) != set([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 3), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)]):
        print "getEmptyCells() not OK."
        print "Expected: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 3), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)]"
        print "Received: ", board.getEmptyCells()     
        checker = 0
        
    print board.draw()
    
    if checker == 1:
        print "Test 2 OK"
    else:
        print "Test 2 not OK"

def test3():
    board = Board(5, 5)
    checker = 1
    cell1 = (2, 3)
    if set(board.getNeighbors(cell1)) != set([(1, 3), (2, 2), (1, 2), (3, 2), (3, 3), (2, 4), (1, 4), (3, 4)]):
        print "getNeighbors() not OK for cell (2, 3)"
        print "Expected: [(1, 3), (2, 2), (1, 2), (3, 2), (3, 3), (2, 4), (1, 4), (3, 4)]"
        print "Received: ", board.getNeighbors(cell1)
        checker = 0
    cell2 = (0, 4)
    if set(board.getNeighbors(cell2)) != set([(0, 3), (1, 3), (1, 4)]):
        print "getNeighbors() not OK for cell (0, 4)"
        print "Expected: [(0, 3), (1, 3), (1, 4)]"
        print "Received: ", board.getNeighbors(cell2)
        checker = 0
    cell3 = (4, 1)
    if set(board.getNeighbors(cell3)) != set([(4, 0), (3, 1), (3, 0), (4, 2), (3, 2)]):
        print "getNeighbors() not OK for cell (4, 1)"
        print "Expected: [(4, 0), (3, 1), (3, 0), (4, 2), (3, 2)]"
        print "Received: ", board.getNeighbors(cell3)        

    print board.draw()
    if checker == 1:
        print "Test 3 OK"
    else:
        print "Test 3 not OK"

def test4():
    game = Game(5, 5, 2, 4)
    checker = 1
    if game.getBoard().getWidth() != 5 or game.getBoard().getHeight() != 5:
        print "getBoard() not OK. Please check"
        checker = 0
    if not game.isActive():
        print "isActive() not OK."
        print "Expected: True"
        print "Received: ", game.isActive()
        checker = 0
    if game.getTurn() != 2:
        print "getTurn() not OK."
        print "Expected: 2"
        print "Received: ", game.getTurn()
        checker = 0
    if game.getWinner() != 0:
        print "getWinner() not OK."
        print "Expected: 0"
        print "Received: ", game.getWinner()
        checker = 0
    game.deactivate()
    if game.isActive():
        print "isActive() or deactivate() not OK."
        print "Expected: False"
        print "Received: ", game.isActive()
        checker = 0
    game.activate()
    if not game.isActive():
        print "isActive() or activate() not OK."
        print "Expected: True"
        print "Received: ", game.isActive()
        checker = 0
    game.switchTurn()
    if game.getTurn() != 1:
        print "switchTurn() not OK."
        print "Expected: 1"
        print "Received: ", game.getTurn()
        checker = 0
    game.declareWinner(2)
    if game.getWinner() != 2:
        print "declareWinner() not OK."
        print "Expected: 2"
        print "Received: ", game.getWinner()
        checker = 0        
    
    game.draw()
    if checker == 1:
        print "Test 4 OK"
    else:
        print "Test 4 not OK"
        
def test5():
    print "Creating a game 5x5. Player O go first."
    game = Game(5, 5, 2, 4)
    checker = 1
    
    try:
        print "Player X wants to mark at (0, 0)"
        game.mark(1, (0, 0))
    except:
        pass
    
    if game.getBoard().getCellStatus((0, 0)) == 1:
        print "mark() not OK."
        print "Player X should not have right to mark, as it's not his turn."
        checker = 0
    
    try:
        print "Player O marks at (1, 1)"
        game.mark(2, (1, 1))
        game.draw()
    except:
        print "mark() not OK. Please check your code."
        return
    
    if game.getBoard().getCellStatus((1, 1)) != 2:
        print "mark() not OK. Please check your code."
        checker = 0
    
    try:
        print "Player X wants to mark at (1, 1)"
        game.mark(1, (1, 1))
    except:
        pass

    if game.getBoard().getCellStatus((1, 1)) == 1:
        print "mark() not OK."
        print "Player X should not have right to mark, as the cell is already occupied."
        checker = 0
    
    print "Try deactivating the game"
    game.deactivate()
    
    try:
        print "Player X wants to mark at (1, 2)"
        game.mark(1, (1, 2))
    except:
        pass
    
    if game.getBoard().getCellStatus((1, 2)) == 1:
        print "mark() not OK."
        print "Player X should not have right to mark, as the game has finished."
        checker = 0
        
    print "Reactivating the game"
    game.activate()
    if game.getTurn() == 2:
        game.switchTurn()
    
    try:
        print "Player X wants to mark at (1, 2)"
        game.mark(1, (1, 2))
        game.draw()
    except:
        pass

    if game.getBoard().getCellStatus((1, 2)) != 1:
        print "mark() not OK. Please check your code."
        checker = 0    
    
    if checker == 1:
        print "Test 5 OK"
    else:
        print "Test 5 not OK"

def test6():
    game = Game(5, 5, 2, 4)
    checker = 1
    game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [2, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 2, 0], [0, 1, 0, 0, 2]])
    if not game.isVictoryCell((3, 1)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(3, 1) should be a victory cell."
        checker = 0
    if game.isVictoryCell((0, 4)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(0, 4) should not be a victory cell."
        checker = 0
    game.draw()
    
    game = Game(5, 5, 2, 4)
    game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [2, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
    if not game.isVictoryCell((1, 4)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(1, 4) should be a victory cell."
        checker = 0
    if game.isVictoryCell((2, 0)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(2, 0) should not be a victory cell."
        checker = 0
    game.draw()
    
    game = Game(5, 5, 2, 4)
    game.getBoard().setBoardStatus([[0, 0, 2, 0, 0], [1, 2, 1, 1, 1], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
    if not game.isVictoryCell((3, 3)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(3, 3) should be a victory cell."
        checker = 0
    if game.isVictoryCell((0, 0)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(0, 0) should not be a victory cell."
        checker = 0
    game.draw()
    
    game = Game(5, 5, 2, 4)
    game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 2, 1, 0], [0, 2, 1, 0, 0]])
    if not game.isVictoryCell((4, 1)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(4, 1) should be a victory cell."
        checker = 0
    if game.isVictoryCell((2, 2)):
        print "isVictoryCell() not OK."
        game.draw()
        print "(2, 2) should not be a victory cell."
        checker = 0
    game.draw()
        
    if checker == 1:
        print "Test 6 OK"
    else:
        print "Test 6 not OK"

def test7():
    game = Game(5, 5, 2, 4)
    checker = 1
    game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 2, 1, 0], [0, 2, 1, 0, 0]])
    if game.isFull():
        game.draw()
        print "isFull() not OK."
        print "Expected: False"
        print "Received: ", game.isFull()
        checker = 0
    game.draw()
    
    game = Game(5, 5, 2, 4)
    game.getBoard().setBoardStatus([[1, 2, 1, 2, 2], [2, 1, 1, 1, 2], [2, 1, 1, 2, 1], [2, 2, 2, 1, 1], [1, 2, 1, 1, 2]])
    if not game.isFull():
        game.draw()
        print "isFull() not OK."
        print "Expected: True"
        print "Received: ", game.isFull()
        checker = 0   
    game.draw()
        
    if checker == 1:
        print "Test 7 OK"
    else:
        print "Test 7 not OK"

def test8():
    game = Game(5, 5, 2, 4)
    checker = 1
    game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 0, 1, 0], [0, 2, 1, 0, 0]])

    if game.getTurn() != 2:
        game.switchTurn()
        
    game.mark(2, (3, 2)) 
    game.draw()
    if game.isActive():
        print "mark() not OK. You may have forgotten to deactivate the game."
        print "Expected: Status active = True"
        print "Received: Status active = ", game.isActive()
        checker = 0
    if game.getWinner() != 2:
        print "mark() not OK. You may have forgotten to declare the winner."
        print "Expected: Winner 2"
        print "Received: Winner ", game.getWinner()
        
    game = Game(5, 5, 2, 4)
    checker = 1
    game.getBoard().setBoardStatus([[0, 2, 1, 2, 2], [2, 1, 1, 1, 2], [1, 1, 2, 2, 1], [2, 2, 1, 1, 1], [1, 2, 1, 1, 2]])

    if game.getTurn() != 1:
        game.switchTurn()
    game.mark(1, (0, 0))

    game.draw()
    if game.isActive():
        print "mark() not OK. You may have forgotten to deactivate the game."
        print "Expected: Status active = True"
        print "Received: Status active = ", game.isActive()
        checker = 0
    if game.getWinner() != 0:
        print "mark() not OK. The game should have no winner."
        print "Expected: Winner 0"
        print "Received: Winner ", game.getWinner()
        
    game = Game(5, 5, 2, 4)
    checker = 1
    game.getBoard().setBoardStatus([[0, 0, 0, 0, 0], [0, 0, 1, 1, 2], [0, 0, 0, 2, 0], [0, 0, 0, 1, 0], [0, 2, 1, 0, 0]])
    
    if game.getTurn() != 2:
        game.switchTurn()
    game.mark(2, (3, 4))
    
    game.draw()
    if not game.isActive():
        print "mark() not OK. "
        print "Expected: Status active = True"
        print "Received: Status active = ", game.isActive()
        checker = 0
    if game.getWinner() != 0:
        print "mark() not OK. The game should have no winner."
        print "Expected: Winner 0"
        print "Received: Winner ", game.getWinner()
    if game.getTurn() != 1:
        print "mark() not OK. You may have forgotten to switch the turn."
        print "Expected: Turn of 1"
        print "Received: Turn of ", game.getTurn()
        
    if checker == 1:
        print "Test 8 OK"
    else:
        print "Test 8 not OK"

def test9():
    X, O = 1, 2
    game = Game(5, 5, O, 4)
    checker = 1
    game.mark(O, (2, 1))
    game.mark(X, (2, 2))
    game.mark(O, (1, 1))
    game.mark(X, (0, 3))
    game.mark(O, (3, 1))
     
    cell_list = []
    for i in range(20):
        cell = game.getBoard().getARandomEmptyCell()
        if cell not in cell_list:
            cell_list.append(cell)
    if len(cell_list) <= 1:
        print "getARandomEmptyCell() not OK. It is not random."
        checker = 0
    if (2, 1) in cell_list or (2, 2) in cell_list or (1, 1) in cell_list or (0, 3) in cell_list or (3, 1) in cell_list:
        print "getARandomEmptyCell() not OK. Some non-empty cell was chosen: (2, 1) or (2, 2) or (1, 1) or (0, 3) or (3, 1)"
        checker = 0
    
    game.mark(X, (4, 0))
    game.mark(O, (4, 1))
    if game.evaluateLastStepRandomly(O) != 1:
        print "evaluateLastStepRandomly() not OK. It should be 1. Received: ", game.evaluateLastStepRandomly(O)

    game.draw()
    if checker == 1:
        print "Test 9 OK"
    else:
        print "Test 9 not OK"

def test10():
    X, O = 1, 2
    game = Game(5, 5, O, 4)
    checker = 1
    game.mark(O, (2, 1))
    game.mark(X, (2, 2))
    game.mark(O, (1, 2))
    game.mark(X, (0, 3))
    game.mark(O, (1, 1))
    
    copy_game = game.generateGameCopy()
    if game == copy_game:
        print "generateGameCopy() not OK. Expected: a copy, received: the instance itself."
        checker = 0
    if game.getBoard() == copy_game.getBoard():
        print "generateGameCopy() not OK. The copy and the main game are sharing the same board. Please use another board for the copy."
        checker = 0
    if not copy_game.isActive():
        print "generateGameCopy() not OK. It is not active while the main game is."
        checker = 0
    if copy_game.getTurn() != game.getTurn():
        print "generateGameCopy() not OK. Please review __turn."
        checker = 0
    if copy_game.getWinner() != game.getWinner():
        print "generateGameCopy() not OK. Please review __winner."
        checker = 0
    if copy_game.getBoard().getBoardStatus() != game.getBoard().getBoardStatus():
        print "generateGameCopy() not OK. The board status are not the same."
        print "Expected: ", game.getBoard().getBoardStatus()
        print "Received: ", copy_game.getBoard().getBoardStatus()
        checker = 0
    
    game.draw()
    if checker == 1:
        print "Test 10 OK"
    else:
        print "Test 10 not OK"
    

def test11():
    print "Launching test11. It may take time. Begin with small number of N (number of copies of the game)"
    checker = 1
    HUMAN, MACHINE = 1, 2
    
    t = time.time()
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    nextStep = game.decideNextStep(MACHINE, Game.algo1)
    print "Next step chosen by machine: ", nextStep
    game.mark(MACHINE, nextStep)
    game.draw()
    
    if nextStep != (3, 1):
        print "Expected: ", (3, 1)
        print "Received: ", nextStep
        print "Please increase the number of copies, or review your implementarion."
        checker = 0
    
    time_taken = time.time() - t
    print "Learning with algo 1 took ", time_taken, " seconds."
    
    if checker == 1:
        print "Test 11 OK"
    else:
        print "Test 11 not OK"
    return time_taken


def test12():
    print "Launch test11 again for comparison"
    time_algo1 = test11()  
    
    print "Launching test12. It may take time. Begin with small number of N (number of copies of the game)"
    checker = 1
    HUMAN, MACHINE = 1, 2
    
    t = time.time()
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    nextStep = game.decideNextStep(MACHINE, Game.algo2)
    print "Next step chosen by machine: ", nextStep
    game.mark(MACHINE, nextStep)
    game.draw()
    
    if nextStep != (3, 1):
        print "Expected: ", (3, 1)
        print "Received: ", nextStep
        print "Please increase the number of copies, or review your implementarion."
        checker = 0
    
    time_taken = time.time() - t
    print "Learning with algo 2 took ", time_taken, " seconds."
    
    if time_taken > time_algo1:
        print "Warning: the algo is implemented but something wrong for runtime. It should run in less than time needed for algo1. Try increasing the number of copies."
    else:
        print "Algo 2 run in ", round(time_taken / time_algo1, 3)*100, "% of time needed of algo 1."
        
    if checker == 1:
        print "Test 12 OK"
    else:
        print "Test 12 not OK"

def test13():
    print "Launch test11 again for comparison"
    time_algo1 = test11()  
    
    print "Launching test13. It may take time."
    checker = 1
    HUMAN, MACHINE = 1, 2
    
    t = time.time()
    game = Game(5, 5, MACHINE, 4)
    game.mark(MACHINE, (2, 1))
    game.mark(HUMAN, (2, 2))
    game.mark(MACHINE, (1, 2))
    game.mark(HUMAN, (0, 3))
    game.mark(MACHINE, (1, 1))
    game.mark(HUMAN, (1, 0))
    nextStep = game.decideNextStep(MACHINE, Game.algo3)
    print "Next step chosen by machine: ", nextStep
    game.mark(MACHINE, nextStep)
    game.draw()
    
    if nextStep != (3, 1):
        print "Expected: ", (3, 1)
        print "Received: ", nextStep
        print "Please increase the number of copies, or review your implementarion."
        checker = 0
    
    time_taken = time.time() - t
    print "Learning with algo 3 took ", time_taken, " seconds."
    
    if time_taken > time_algo1:
        print "Warning: the algo is implemented but something wrong for runtime. It should run in less than time for algo1."
    else:
        print "Algo 3 run in ", round(time_taken / time_algo1, 3)*100, "% of time needed of algo 1."
        
    if checker == 1:
        print "Test 13 OK"
    else:
        print "Test 13 not OK"
        
Tests = [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13]

#MAIN FUNCTION
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please configure your test by Run -> Configure"
    else:
        for i in range(len(Tests) + 1):
            if sys.argv[1] == "test_" + str(i):
                Tests[i]()
                break