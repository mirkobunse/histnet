import argparse
import numpy as np
import torch
from dlquantification.quantmodule.histograms.HardHistogramBatched import HardHistogramBatched
from dlquantification.quantmodule.transformers.modules import MAB, ISAB, PMA
from sklearn.datasets import make_classification

def main(output_path, module):
  X, y = make_classification( # generate data for testing
    n_classes = 2,
    n_features = 5,
    n_informative = 3,
    n_samples = 2000,
    random_state = 25,
  )
  X = X.astype(np.float32)

  if module == "histnet": # generate tests for the histogram layer
    cases = [ # two different inputs and two different n_bins
      { "input": X[y == 0], "n_bins": 4 },
      { "input": X[y == 1], "n_bins": 4 },
      { "input": X[y == 0], "n_bins": 8 },
      { "input": X[y == 1], "n_bins": 8 },
    ]
    for c in cases: # generate desired outputs
      m = HardHistogramBatched(
        n_features = c["input"].shape[1],
        num_bins = c["n_bins"]
      )
      c["output"] = m.forward(torch.from_numpy(c["input"])).detach().numpy()

  elif module == "mab": # generate tests for the SetTransformer's MAB module
    cases = [
      { "X1": X[y == 0], "X2": X[y == 1], "n_features_per_head": 2, "n_heads": 4 },
      { "X1": X[y == 1], "X2": X[y == 0], "n_features_per_head": 3, "n_heads": 3 },
    ]
    for c in cases:
      m = MAB(
        dim_Q = c["X1"].shape[1],
        dim_K = c["X2"].shape[1],
        dim_V = c["n_features_per_head"] * c["n_heads"],
        num_heads = c["n_heads"],
      )
      c["output"] = m.forward(
        torch.from_numpy(c["X1"]).unsqueeze(0),
        torch.from_numpy(c["X2"]).unsqueeze(0),
      ).squeeze(0).detach().numpy()

  elif module == "isab": # generate tests for the SetTransformer's ISAB module
    cases = []
    pass # TODO

  elif module == "pma": # generate tests for the SetTransformer's PMA module
    cases = []
    pass # TODO

  else:
    raise ValueError(f"Unknown module=\"{module}\"")

  # store the results
  np.save(output_path, cases)
  print(f"Stored {len(cases)} unittest cases at {output_path}")

# command line interface
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("output_path", help="path of an output *.npy file", type=str)
  parser.add_argument("--module", help="which module to test", type=str, required=True)
  args = parser.parse_args()
  main(args.output_path, args.module)
