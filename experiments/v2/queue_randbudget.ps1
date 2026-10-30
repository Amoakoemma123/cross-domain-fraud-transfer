# Label-blind budget sensitivity: ULB target, 1% budget, transfer + target-only, 5 seeds x 8 draws
Set-Location E:\Research_Works\Fraud_Detection_Research\financial-fraud-transfer
foreach ($r in 0..7) {
  foreach ($s in 42,123,456,789,1024) {
    & .\fraud_env\Scripts\python.exe -u -m experiments.v2.run_v2 --seed $s --tgt ulb --fracs 0.01 --random-budget --budget-rep $r --methods tgt_rf tgt_lgbm tgt_mlp pooled_lgbm tradaboost finetune contrastive *> "experiments\v2\run_rand_s$($s)_r$r.log"
  }
}
