import logging

import jsonlines
import pandas as pd
from datasets import load_metric
from rouge import Rouge
from tqdm import tqdm


def use_datasets(generated_summaries, reference_summaries):
    metric_result = {}
    metric = load_metric("rouge", seed=42)

    result = metric.compute(predictions=generated_summaries, references=reference_summaries, use_stemmer=True)

    # result = metric.compute(predictions=['add http server for frontend and snapshots'], references=['use http server instead of interception'])

    # Extract a few results from ROUGE
    precisions = {key: value.mid.precision * 100 for key, value in result.items()}
    precisions = {k: round(v, 2) for k, v in precisions.items()}
    metric_result["precision"] = precisions

    recalls = {key: value.mid.recall * 100 for key, value in result.items()}
    recalls = {k: round(v, 2) for k, v in recalls.items()}
    metric_result["recall"] = recalls

    fmeasures = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    fmeasures = {k: round(v, 2) for k, v in fmeasures.items()}
    metric_result["f1"] = fmeasures

    return metric_result


def use_rouge(generated_summaries, reference_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    print(scores)
    return scores


class Metric:
    @staticmethod
    def rouge_compete(compete_path: str, output_path: str, ) -> None:

        generated_summaries = []
        reference_summaries = []

        chunk = []
        with jsonlines.open(compete_path, "r") as reader:
            for i, line in enumerate(reader):
                chunk.append(line)
        df = pd.DataFrame(chunk)

        for i, row in tqdm(df.iterrows(), "Processing prompts from input file"):
            if row['Prediction'] is not None:
                generated_summaries.append(row['Prediction'])
                reference_summaries.append(row['Title'])

        with jsonlines.open(output_path, "w") as file:
            result = use_datasets(generated_summaries, reference_summaries)
            total_length = 0
            for string in generated_summaries:
                string = string.split(" ")
                total_length += len(string)
            result["avg_length"] = total_length / len(generated_summaries)
            file.write(result)
        logging.info("Finish Metric Result!")
