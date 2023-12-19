
board_cell_size = 30 # 単位[mm]

class RobotSystem:
    
    def arm_control(self, y, x):
        revese = False
        y = y * board_cell_size
        x = x * board_cell_size
        
        return revese, y, x