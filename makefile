
CK.zip: *.py *.pyz
	zip CK.zip evaluate.py *.pyz example_solutions/example_general_discrete_problem_training.py example_solutions/example_classical_svm_training.py example_solutions/example_problem_zero_training.py example_solutions/helper_functions.py


%_spec.pyz: problem_spec_script.py
	python problem_spec_script.py --problems $(word 1,$(subst _, ,$@))  


