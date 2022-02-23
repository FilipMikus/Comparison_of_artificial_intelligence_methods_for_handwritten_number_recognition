from model import Model
from view import GraphicalUserInterface


class ImagePreprocessingApp:
    def __init__(self):
        model = Model()
        gui = GraphicalUserInterface(model)
        model.gui = gui
        gui.open_gui_window()


if __name__ == '__main__':
    ImagePreprocessingApp()
