import typer
import cv2
from pathlib import Path

app = typer.Typer()

MIN_DIM = 300
def del_if_small(file:Path):
    img = cv2.imread(str(file))
    h, w = img.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        print(f'Deleting {file} {h}x{w}')
        file.unlink()

@app.command()
def del_small_files():
    for file in Path('data').rglob('*.jpg'):
        del_if_small(file)

@app.command()
def nop():
    pass

if __name__ == "__main__":
    app()