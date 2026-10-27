# Training runs

Runs are numbered in chronological order. Original timestamp IDs are retained below and in `run_rename_map.json`. Detailed settings and scores are in `run_index.csv`.

| Run | Original ID | Test accuracy |
|---|---|---:|
| [1 - Baseline with random oversampling](<1 - Baseline with random oversampling>) | `20261026_104526_006227` | 41.73% |
| [2 - SMOTE oversampling](<2 - SMOTE oversampling>) | `20261026_105105_296680` | 42.27% |
| [3 - No oversampling baseline](<3 - No oversampling baseline>) | `20261026_105253_335505` | 42.40% |
| [4 - Class-weighted cross entropy](<4 - Class-weighted cross entropy>) | `20261026_112624_335924` | 40.93% |
| [5 - Dropout 0.20 without class weights](<5 - Dropout 0.20 without class weights>) | `20261026_141137_046468` | 43.20% |
| [6 - Dropout 0.10](<6 - Dropout 0.10>) | `20261026_145558_319099` | 40.53% |
| [7 - Dropout 0.05](<7 - Dropout 0.05>) | `20261026_145620_602247` | 42.13% |
| [8 - Exclude Medium-High and High from training](<8 - Exclude Medium-High and High from training>) | `20261026_153815_845964` | 40.80% |
| [9 - Patience 2 baseline](<9 - Patience 2 baseline>) | `20261026_154313_647721` | 40.80% |
| [10 - Larger hidden layers 64-32 with dropout 0.05](<10 - Larger hidden layers 64-32 with dropout 0.05>) | `20261026_162535_699665` | 45.73% |
| [11 - Repeat run 10 with training seed 42](<11 - Repeat run 10 with training seed 42>) | `20261026_162854_068037` | 45.73% |
| [12 - Training seed 43 with fixed data seed 42](<12 - Training seed 43 with fixed data seed 42>) | `20261026_163117_857577` | 38.00% |
| [13 - Restore all five training classes](<13 - Restore all five training classes>) | `20261026_163315_526749` | 39.73% |
| [14 - Two-class diagnostic Medium-Low vs Medium](<14 - Two-class diagnostic Medium-Low vs Medium>) | New numbered run | 43.87% (two-output model) |
| [15 - Memorize 64 balanced samples without dropout or L2](<15 - Memorize 64 balanced samples without dropout or L2>) | New numbered run | Not evaluated (training only) |
| [16 - Hidden 32 dropout 0.05 patience 2 seed 43 - all classes](<16 - Hidden 32 dropout 0.05 patience 2 seed 43 - all classes>) | New numbered run | 44.80% |
| [17 - Hidden 16 dropout 0.05 patience 2 seed 43 - all classes](<17 - Hidden 16 dropout 0.05 patience 2 seed 43 - all classes>) | New numbered run | 40.80% |
| [18 - Hidden 32 dropout 0.05 patience 10 seed 43 - all classes](<18 - Hidden 32 dropout 0.05 patience 10 seed 43 - all classes>) | New numbered run | 43.87% |
| [19 - Hidden 32 dropout 0.05 patience 5 seed 43 - all classes](<19 - Hidden 32 dropout 0.05 patience 5 seed 43 - all classes>) | New numbered run | 43.87% |
| [20 - Hidden 32 dropout 0.05 patience 5 lr 0.0005 seed 43 - all classes](<20 - Hidden 32 dropout 0.05 patience 5 lr 0.0005 seed 43 - all classes>) | New numbered run | 43.73% |
| [21 - PReLU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes](<21 - PReLU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes>) | New numbered run | 42.80% |
| [22 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes](<22 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes>) | New numbered run | 44.27% |
| [23 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets](<23 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets>) | New numbered run | 44.27% |
| [24 - L1 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets](<24 - L1 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets>) | New numbered run | 43.87% |
| [25 - L2 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets](<25 - L2 - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets>) | New numbered run | 44.27% |
| [26 - ElasticNet - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets](<26 - ElasticNet - GELU hidden 32 dropout 0.05 patience 5 lr 0.001 seed 43 - all classes - one-hot targets>) | New numbered run | 44.27% |

Run 14 is a two-class validation diagnostic: validation accuracy **53.12%**, majority baseline **51.52%**. Its saved two-output model was subsequently evaluated on all 750 original five-class test examples: **43.87% accuracy**, versus **43.07%** for the majority baseline. This separate evaluation is in `full_five_class_test_evaluation`; three true classes are outside the model outputs and always count as errors.

Existing plot sidebars retain the original timestamp IDs. Comparison CSVs and JSON reports refer to the renamed folders.

New runs use the next available number followed by a description of their settings.

Run 15 is a training-only sanity check: **64/64 examples memorized**, first perfect accuracy at epoch **111**, final cross entropy **0.00965** at epoch **292**. No validation or test score is reported. This is not evidence of generalization.
