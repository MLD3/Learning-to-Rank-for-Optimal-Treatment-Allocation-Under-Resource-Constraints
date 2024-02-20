** Code for Learning to Rank for Optimal Treatment Allocation Under Resource Constraints **

Code for Learning to Rank for Optimal Treatment Allocation Under Resource Constraints, published at AIStats 2024. 

Instructions: 
1. Install required packages in `requirements.txt`
2. Set relevant hyperparameters in `run_models.py`. 
	a. `method = 1` runs a random forest splitting based on AUTOC, whereas `method = 0` runs a baseline model that splits to maximize MSE. 
	b. Other relevant hyperparameters are 
3. Run `run_models.py`
	a. Models and results will be saved out in relevant directories, with unique identifiers based on relevant parameters. These can be adjusted as needed. 