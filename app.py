import torch
import torch.nn as nn
import gradio as gr
import torchvision.transforms as T
from PIL import Image

# ── FractalSIE Architecture (your mathematical rule) ──
class SpiralBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw   = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.pw   = nn.Conv2d(C, C, 1)
        self.norm = nn.GroupNorm(8, C)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x)))) + x

class FractalSIE(nn.Module):
    def __init__(self, C=192, n_layers=16, n_cls=10, drop_p=0.0):
        super().__init__()
        self.stem  = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1),
            nn.GroupNorm(8, C), nn.GELU()
        )
        self.block = SpiralBlock(C)
        self.down  = nn.Sequential(
            nn.Conv2d(C, C*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, C*2), nn.GELU()
        )
        self.head  = nn.Linear(C*2, n_cls)
        self.n     = n_layers
    def forward(self, x):
        x = self.stem(x)
        for m in range(self.n):
            gate = (2**m) / (2**m + 1)  # Your doubling rule
            x    = self.block(x) * gate
        return self.head(self.down(x).mean([-1,-2]))

# ── Load your trained model ──
CLASSES = ['Airplane','Car','Bird','Cat','Deer',
           'Dog','Frog','Horse','Ship','Truck']

model = FractalSIE(C=192, n_layers=16, n_cls=10, drop_p=0.0)
model.load_state_dict(
    torch.load('sie_best.pt', map_location='cpu')
)
model.eval()

# ── Image preprocessing ──
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

# ── Prediction function ──
def predict(image):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(10)}

# ── Gradio Interface ──
gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Upload any image'),
    outputs=gr.Label(num_top_classes=3, label='Predictions'),
    title='🌀 SIE — Spiral Intelligence Engine',
    description=(
        '**Abdullahi Abdullahi Dantala** | Independent Researcher, Abuja, Nigeria\n\n'
        'This model achieves **87.44% accuracy** on CIFAR-10. '
        'Architecture derived from a hand-drawn spiral using the cyclic Pascal recurrence: '
        'P_(m+1,n) = P_(m,n) + P_(m,(n mod 12)+1)\n\n'
        'Upload an image of: airplane, car, bird, cat, deer, dog, frog, horse, ship, or truck.'
    ),
    examples=[],
    theme=gr.themes.Soft()
).launch()
