python train_lequa.py -t settransformers_T1B -n settransformers -p ./parameters/settransformers_T1B.json -f rff -d T1B -c cuda:0 > output/T1B_settransformers.out  &
python train_lequa.py -t deepsets_max_T1B -n deepsets -p ./parameters/deepsets_max.json -f rff -d T1B -c cuda:1 > output/T1B_deepsets_max.out  &
wait
python train_lequa.py -t deepsets_avg_T1B -n deepsets -p ./parameters/deepsets_avg.json -f rff -d T1B -c cuda:0 > output/T1B_deepsets_avg.out  &
python train_lequa.py -t deepsets_median_T1B -n deepsets -p ./parameters/deepsets_median.json -f rff -d T1B -c cuda:1 > output/T1B_deepsets_median.out  &
wait
python train_lequa.py -t histnet_sigmoid_T1B -n histnet -p ./parameters/histnet_sigmoid_T1B.json -f rff -d T1B -c cuda:0 > output/T1B_histnet_sigmoid.out  &
python train_lequa.py -t histnet_soft_T1B -n histnet -p ./parameters/histnet_soft_T1B.json -f rff -d T1B -c cuda:1 > output/T1B_histnet_soft.out  &
wait
python train_lequa.py -t histnet_softrbf_T1B -n histnet -p ./parameters/histnet_softrbf_T1B.json -f rff -d T1B -c cuda:0 > output/T1B_histnet_softrbf.out  &
python train_lequa.py -t histnet_hard_T1B -n histnet -p ./parameters/histnet_hard_T1B.json -f rff -d T1B -c cuda:1 > output/T1B_histnet_hard.out  &
wait
