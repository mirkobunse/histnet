import argparse
import numpy as np
import torch
from dlquantification.quantmodule.histograms.HardHistogramBatched import HardHistogramBatched
from sklearn.datasets import make_classification

def main(output_path):
  X, y = make_classification( # generate data for testing
    n_classes = 2,
    n_features = 5,
    n_informative = 3,
    n_samples = 2000,
    random_state = 25,
  )
  X = X.astype(np.float32)
  cases = [ # two different inputs and two different n_bins
    { "input": X[y == 0], "n_bins": 4 },
    { "input": X[y == 1], "n_bins": 4 },
    { "input": X[y == 0], "n_bins": 8 },
    { "input": X[y == 1], "n_bins": 8 },
  ]
  for c in cases: # generate desired outputs
    module = HardHistogramBatched(
      n_features = c["input"].shape[1],
      num_bins = c["n_bins"]
    )
    c["output"] = module.forward(torch.from_numpy(c["input"])).detach().numpy()
  np.save(output_path, cases)
  print(f"Stored {len(cases)} unittest cases at {output_path}")

# command line interface
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("output_path", help="path of an output *.npy file", type=str)
  main(parser.parse_args().output_path)
