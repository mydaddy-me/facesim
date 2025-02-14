from pathlib import Path
import cv2
from tqdm import tqdm
import typer

from facesim import crop_face

app = typer.Typer()

class fs:
    data = Path('data')
    faces = Path('faces')

@app.command()
def faces():
    fs.faces.mkdir(exist_ok=True)
    for f in tqdm(Path('data').rglob('*.jpg')):
        label = f.parent.name
        image = cv2.imread(str(f))
        face = crop_face(image)
        
        if face is None:
            continue
        
        h, w = face.shape[:2]
        
        if(min(h, w) < 256):
            continue

        out_dir = fs.faces / label
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f.name
        cv2.imwrite(str(out_file), face)

@app.command()
def nop():
    pass

if __name__ == "__main__":
    app()