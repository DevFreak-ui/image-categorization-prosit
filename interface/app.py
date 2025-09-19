# app.py
from flask import Flask, request, render_template_string
from PIL import Image
import io, torch, torch.nn as nn
import torchvision.transforms as T

# ---------- Model definition must match training ----------
class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)             # logits
        return x

# ---------- Config ----------
MODEL_WEIGHTS = "checkpoints/dnn_cinic10_state_dict.pth"  # your saved state_dict
CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

tfm = T.Compose([T.Resize((32,32)), T.ToTensor(), T.Normalize(MEAN, STD)])

device = torch.device("cpu")

# ---------- Load model ----------
model = SimpleDNN().to(device)
state = torch.load(MODEL_WEIGHTS, map_location=device)
model.load_state_dict(state)
model.eval()

# ---------- Minimal UI ----------
TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"><title>CINIC-10 DNN Demo</title>
  <style> body{font-family:system-ui,Arial;margin:2rem auto;max-width:700px}
  .card{border:1px solid #ddd;padding:1rem;border-radius:.5rem} pre{white-space:pre-wrap}</style>
</head>
<body>
  <h2>CINIC-10 DNN Demo (Flask)</h2>
  <div class="card">
    <form method="post" enctype="multipart/form-data" action="/predict">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Predict</button>
    </form>
  </div>
  {% if result %}
    <h3>Top-5 Predictions</h3>
    <pre>{{ result }}</pre>
  {% endif %}
</body>
</html>
"""

app = Flask(__name__)

@app.get("/")
def home():
    return render_template_string(TEMPLATE)

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return render_template_string(TEMPLATE, result="Error: no file provided.")
    f = request.files["file"]
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        return render_template_string(TEMPLATE, result="Error: invalid image.")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)                     # raw scores
        probs = torch.softmax(logits, dim=1)  # softmax only for display
        topk = torch.topk(probs, k=5)
    result_lines = []
    for rank, (idx, p) in enumerate(zip(topk.indices[0].tolist(), topk.values[0].tolist()), start=1):
        result_lines.append(f"{rank}. {CLASS_NAMES[idx]} — {p:.4f}")
    return render_template_string(TEMPLATE, result="\n".join(result_lines))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
