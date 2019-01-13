from flask import Flask, jsonify, request
import torch
from game2048.displays import Display

def get_flask_app(game, agent):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return app.send_static_file('board.html')

    @app.route("/board", methods=['GET', 'POST'])
    def get_board():
        direction = -1
        control = "USER"
        if request.method == "POST":
            direction = request.json
            if direction == -1:
                direction = agent.step()
                control = 'AGENT'
            game.move(direction)
        return jsonify({"board": game.board.tolist(),
                        "score": game.score,
                        "end": game.end,
                        "direction": direction,
                        "control": control})

    return app


if __name__ == "__main__":
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    APP_PORT = 5005
    APP_HOST = "localhost"

    from game2048.game import Game
    from game2048.Model import Net

    model = Net()
    model.load_state_dict(torch.load("./game2048/para.pkl", map_location='cpu'))
    model.eval()

    game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)
    from game2048.agents import LXYAgent

    try:
       # from game2048.agents import ExpectiMaxAgent
       from game2048.agents import LXYAgent
       agent = LXYAgent(model, game, display=Display())
       #agent = LXYAgent(game=game)
    except:
        from game2048.agents import RandomAgent
        print("WARNING: Please compile the ExpectiMaxAgent first following the README.")
        print("WARNING: You are now using a RandomAgent.")
        agent = RandomAgent(game=game)

    print("Run the webapp at http://<any address for your local host>:%s/" % APP_PORT)    
    
    app = get_flask_app(game, agent)
    app.run(port=APP_PORT, threaded=False, host=APP_HOST)  # IMPORTANT: `threaded=False` to ensure correct behavior
    
    
