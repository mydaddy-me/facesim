import typer
import cv2
from pathlib import Path

app = typer.Typer()

MIN_DIM = 350
def del_if_small(file:Path):
    img = cv2.imread(str(file))
    h, w = img.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        print(f"Deleting {file} because it's too small")
        file.unlink()

@app.command()
def del_small_files():
    for file in Path('data').rglob('*.jpg'):
        del_if_small(file)

if __name__ == "__main__":
    app()