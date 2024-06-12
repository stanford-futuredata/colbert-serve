PROCS=1
lambda_values=("1.0" "0.05" "0.1" "0.2" "0.3" "0.5" "0.7" "0.9")
exps=("search" "serve" "pisa")
index="wiki"

# Loop through the lambda values and run the command
for lambda in "${lambda_values[@]}"; do
	for exp in "${exps[@]}"; do
	    input_file="traces/trace_lam_${lambda}.txt"
	    mkdir -p final_results_${index}/${exp}
	    timing_output="final_results_${index}/${exp}/timing_${lambda}.txt"
	    rankings_output="final_results_${index}/${exp}/rankings_${lambda}.tsv"
	    
	    python driver.py -n 1 -w 1 -i $index -t $input_file -o $timing_output -r $rankings_output -e $exp
	    echo "Finished evaluating $exp for lambda = $lambda"
	done
done
