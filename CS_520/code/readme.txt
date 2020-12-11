=== Evaluation Code ===

To run eval_lingam.py, eval_rl.py, or eval_gnn.py, first set the appropriate source path in the code.  After running, a csv file will be generated with the results.


=== For run_lingam.py ===

Required packages:

lingam ( install from https://github.com/cdt15/lingam )
numpy
pandas
graphviz

Usage:

python lingam/run_lingam.py -d <source_data_path> -s <save_results_path>

source_data_path must have 2 numpy arrays in it: data.npy and DAG.npy


=== For Causal_Discovery_RL ===

See https://github.com/huawei-noah/trustworthyAI/tree/master/Causal_Structure_Learning/Causal_Discovery_RL