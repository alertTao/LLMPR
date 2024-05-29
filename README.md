# Structure
- Data
- Installing Necessary Libraries
- Configuration File
- Matching Methods
- Generation Methods
- Results


## Data
The data is located in [`./data`](./data), 
where BM25 and TF-idf represent the most similar matching records of the two algorithms (1-6). 
"no-token" and "with-token" are the datasets collected by Zhang et al., while "test.jsonl" standardizes the data into JSON format.

## Installing Necessary Libraries
```
pip install -r requirements.txt
pip install hydra-core --upgrade
```

## Configuration File
Configure the experiment in[`./conf/config.yaml`](./conf/config.yaml), 
including settings for LLMs generation, GPU device selection, 
LLMs selection (model_id), matching method (matching), 
and number of most similar records to select (shot).

## Matching Methods
Includes two methods: BM25 and TF-idf. Run the following command to find the most similar PRs using the corresponding algorithm:
```
python match.py
```
The top 6 most similar PRs from each algorithm on the test dataset of 4382 records can be directly accessed in[`./data/BM25`](./data/BM25)and[`./data/TF-idf`](./data/TF-idf).


## Generation Methods


```
python prompt.py
```



### BlueLM/Llama3
You need to configure your GPU device and add your cache_home in[`./prompt.py`](./prompt.py).

### DeepSeek-V2
You need to add your api_key in[`./source/llm/model.py`](./source/llm/model.py).

## Results
Our execution results are stored in the[`./result`](./result)directory, including results for BlueLM, DeepSeek-v2, and Llama3 LLMs under various conditions of shot (0, 1, 3, 5) and temperature (0, 0.5, 1.0, 1.5, 2.0).

The PRs used for human evaluation are stored in[`./appendix/questionnaire.csv`](./appendix/questionnaire.csv).
