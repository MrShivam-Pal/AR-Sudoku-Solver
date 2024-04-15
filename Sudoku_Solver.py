N = 9

def check_for_non_zero_board(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return False
    return True

def printing(arr):
	for i in range(N):
		for j in range(N):
			print(arr[i][j], end=" ")
		print()

def isSafe(grid, row, col, num):
    
    for x in range(9):
	    if ((grid[row][x] == num) or (grid[x][col] == num) or (grid[3*(row//3) + x//3][3*(col//3) + x%3] == num)):
		    return False
  
    return True

def solveSudoku(grid, row, col):

	if (row == N - 1 and col == N):
		return True

	if col == N:
		row += 1
		col = 0

	if grid[row][col] > 0:
		return solveSudoku(grid, row, col + 1)

	for num in range(1, N + 1, 1):

		if isSafe(grid, row, col, num) == True:

			grid[row][col] = num

			if solveSudoku(grid, row, col + 1):
				return True
		grid[row][col] = 0
  
	return False

def solve_sudoku(grid):
    
    if( solveSudoku(grid , 0 , 0)):
        return grid
    else:
        return None