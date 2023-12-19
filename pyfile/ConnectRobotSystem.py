import json

board_cell_size = 30 # 単位[mm]

class RobotSystem():
    
    def coord_trans(self, y, x):
        y_coord = y * board_cell_size
        x_coord = x * board_cell_size
        return y_coord, x_coord
        
    def format_json(self, y, x, reverse_call):
        data = {
                "y": y,
                "x": x,
                "reverse_call": reverse_call
                }
        json_data = json.dumps(data)
        print(json_data)
        return json_data
    
    def send_data(self, json_data):
        pass