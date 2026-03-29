from Workflow import WorkFlow
import os

if os.name == "nt":
    os.system("cls")
elif os.name == "posix":
    os.system("clear")

def Main():
    WorkFlow()

if __name__ == "__main__":
    Main()
