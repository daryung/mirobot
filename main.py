import tkinter as tk

from src.gui import DrawingRobotApp


def main():
    root = tk.Tk()
    DrawingRobotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
