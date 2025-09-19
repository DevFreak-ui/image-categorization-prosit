# Image Categorization Project
---

### How to run the application

1. Install Dependencies

```Bash
pip install flask torch torchvision pillow
```

2. Run the commnad below in your command prompt

```Bash
python app.py
# open http://127.0.0.1:8000
```
---

### Quick test with 10 samples

```bash
python main.py --test_mode --model_type cnn --model_variant simple
```

### Full training with automatic evaluation
```bash
python main.py --model_type dnn --model_variant vanilla --auto_evaluate
python main.py --model_type cnn --model_variant simple --auto_evaluate
```

### Manual detailed evaluation
```
python evaluate.py --model_path checkpoints/cnn_simple_trained.pth --model_type cnn --model_variant simple --detailed
```

### Statistical evaluation over 5 runs
```bash
python evaluate.py --model_path checkpoints/dnn_vanilla_trained.pth --model_type dnn --model_variant vanilla --num_runs 5
```