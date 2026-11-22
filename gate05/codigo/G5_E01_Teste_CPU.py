import json
import statistics
import time

import numpy
import torch
import torchvision

torch.manual_seed(501)
torch.set_num_threads(4)
model = torchvision.models.resnet18(weights=None).eval()
sample = torch.rand(1, 3, 224, 224)
with torch.inference_mode():
    for _ in range(3):
        output = model(sample)
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        output = model(sample)
        timings.append(1000 * (time.perf_counter() - start))
assert output.shape == (1, 1000) and torch.isfinite(output).all()
print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__,
                      numpy=numpy.__version__, dispositivo=str(output.device),
                      threads=torch.get_num_threads(), mediana_ms=statistics.median(timings),
                      repeticoes=len(timings), entrada=list(sample.shape),
                      saida=list(output.shape)), indent=2))
