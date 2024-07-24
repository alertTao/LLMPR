import logging
import string
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from nltk.translate.meteor_score import meteor_score
import jsonlines
import pandas as pd
from datasets import load_metric
from rouge import Rouge
from tqdm import tqdm

import nltk
import math
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


        bleu = 0
        meteor = 0
        for i in range(len(generated_summaries)):
            print(i)
            print(generated_summaries[i])
            print(reference_summaries[i])
            bleu+=compete_bleu(generated_summaries[i],reference_summaries[i])
            meteor+=compete_meteor(generated_summaries[i],reference_summaries[i])
        with jsonlines.open(output_path, "w") as file:
            result = use_datasets(generated_summaries, reference_summaries)
            total_length = 0
            for string in generated_summaries:
                string = string.split(" ")
                total_length += len(string)
            result["avg_length"] = total_length / len(generated_summaries)
            result['BLEU'] = bleu * 100 / len(generated_summaries)
            result['METEOR'] = meteor * 100 / len(generated_summaries)
            file.write(result)
        # logging.info("Finish Metric Result!")


def compete_bleu(prediction,title):


    reference = title

    candidate = prediction

    # 将句子分词为单词列表
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())

    # 定义平滑函数
    smooth = SmoothingFunction().method1

    # 计算 BLEU-1 到 BLEU-4 分数
    bleu1 = sentence_bleu([reference_tokens], candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.33, 0.33, 0.33, 0),
                          smoothing_function=smooth)
    bleu4 = sentence_bleu([reference_tokens], candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smooth)
    print(bleu4)
    return bleu4

def compete_meteor(prediction,title):

    # 参考翻译和机器翻译的输出（单个句子）
    reference = title
    candidate = prediction

    # 将句子分词为单词列表
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())

    # 计算 METEOR 分数
    score = meteor_score([reference_tokens], candidate_tokens)
    return score

if __name__ == "__main__":
    # 下载需要的 NLTK 数据包
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    # 需要计算的结果的路径
    compete_path = '****'
    output_path = '*****'
    metric = Metric()
    metric.rouge_compete(compete_path,output_path)