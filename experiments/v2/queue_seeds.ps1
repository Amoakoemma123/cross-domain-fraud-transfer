Set-Location E:\Research_Works\Fraud_Detection_Research\financial-fraud-transfer
while (Get-Process -Id 11616 -ErrorAction SilentlyContinue) { Start-Sleep 120 }
foreach ($s in 123,456,789,1024) {
  & .\fraud_env\Scripts\python.exe -u -m experiments.v2.run_v2 --seed $s *> "experiments\v2\run_seed$s.log"
}
