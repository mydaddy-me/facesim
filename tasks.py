from pathlib import Path
import cv2
from tqdm import tqdm
import typer

from facesim import crop_face, crop_parts

app = typer.Typer()

class fs:
    data = Path('data')
    faces = Path('faces')
    nose = Path('nose')
    lips = Path('lips')

@app.command()
def faces():
    fs.faces.mkdir(exist_ok=True)
    for f in tqdm(fs.data.rglob('*.jpg')):
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
def parts():
    fs.nose.mkdir(exist_ok=True)
    fs.lips.mkdir(exist_ok=True)

    for f in tqdm(fs.faces.rglob('*.jpg')):
        label = f.parent.name
        face = cv2.imread(str(f))

        parts = crop_parts(face)
        
        if parts.nose is not None:
            nose_dir = fs.nose / label
            nose_dir.mkdir(exist_ok=True)
            nose_file = nose_dir / f.name

            cv2.imwrite(str(nose_file), parts.nose)

        if parts.lips is not None:
            lips_dir = fs.lips / label
            lips_dir.mkdir(exist_ok=True)
            lips_file = lips_dir / f.name

            cv2.imwrite(str(lips_file), parts.lips)

@app.command()
def nop():
    pass

if __name__ == "__main__":
    app()