import logging
import math
import re
import jsonlines
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """
        BM25算法的构造器
        :param docs: 文档列表，每个文档是一个字符串
        :param k1: BM25算法中的调节参数k1
        :param b: BM25算法中的调节参数b
        """
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc.split()) for doc in docs]  # 计算每个文档的长度
        self.avgdl = sum(self.doc_len) / len(docs)  # 计算所有文档的平均长度
        self.doc_freqs = []  # 存储每个文档的词频
        self.idf = {}  # 存储每个词的逆文档频率
        self.initialize()

    def initialize(self):
        """
        初始化方法，计算所有词的逆文档频率
        """
        df = Counter()
        for doc in self.docs:
            # 为每个文档创建一个词频统计
            self.doc_freqs.append(Counter(doc.split()))
            # 更新df值
            df.update(set(doc.split()))
        # 计算每个词的IDF值
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.docs) - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, doc, query):
        """
        计算文档与查询的BM25得分
        :param doc: 文档的索引
        :param query: 查询字符串
        :return: 该文档与查询的相关性得分
        """
        score = 0.0
        doc_words = self.docs[doc].split()
        for word in query.split():
            if word in doc_words:
                freq = doc_words.count(word)  # 词在文档中的频率
                # 应用BM25计算公式
                score += (self.idf[word] * freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * len(doc_words) / self.avgdl))
        return score

    def highest_scored_sentence(self, query):
        """
        寻找得分最高的句子
        :param query: 查询字符串
        :return: 得分最高的句子和其得分
        """
        max_score = float('-inf')
        position = 0
        for doc_idx, doc in enumerate(self.docs):
            score = self.score(doc_idx, query)
            if score > max_score:
                max_score = score
                position = doc_idx
        return position


class MatchingCaculator:
    def __init__(self,
                 prompt_configuration: str
                 ):
        self._prompt_configuration = prompt_configuration

    def match_data(
            self, source_no_test: str, source_no_train: str, source_train: str, matching: str, output_path: str,
            use_cache: bool
    ) -> None:
        if matching is None:
            logging.info("No Matching Way!")
            return

        logging.info("Start Matching Data!")
        df_no_test = pd.read_csv(source_no_test)
        df_no_train = pd.read_csv(source_no_train)
        df_train = pd.read_csv(source_train)

        print(len(df_no_test))
        print(len(df_no_train))
        print(len(df_train))
        docs = df_no_test['text']
        sources = df_no_train['text']
        sources_text = df_train['text']
        sources_title = df_train['summary']

        if matching == 'BM25':
            bm25 = BM25(sources)
            with jsonlines.open(output_path, "w") as file:
                for i in range(len(docs)):
                    if i % 10 == 0:
                        print(i)
                    position = bm25.highest_scored_sentence(docs[i])
                    description = ""
                    commit_messages = ""
                    linked_issue_titles = ""
                    desc_content = re.findall(r'<desc>(.*?)</desc>', sources_text[position], re.DOTALL)
                    cmt_content = re.findall(r'<cmt>(.*?)</cmt>', sources_text[position], re.DOTALL)
                    iss_content = re.findall(r'<iss>(.*?)</iss>', sources_text[position], re.DOTALL)
                    for desc in desc_content:
                        description += desc
                    for cmt in cmt_content:
                        commit_messages += cmt
                    for iss in iss_content:
                        linked_issue_titles += iss

                    pre_data = {
                        "description": description,
                        "commit_messages": commit_messages,
                        "linked_issue_titles": linked_issue_titles,
                        "title": sources_title[position]
                    }
                    file.write(pre_data)
            logging.info("Finish Matching Data!")
        elif matching == 'TF-idf':

            # 创建TF-idf向量化器
            tfidf_vectorizer = TfidfVectorizer()

            # 计算TF-idf向量
            tfidf_corpus = tfidf_vectorizer.fit_transform(sources)

            with jsonlines.open(output_path, "w") as file:
                for i in range(len(docs)):
                    if i % 10 == 0:
                        print(i)
                    tfidf_query = tfidf_vectorizer.transform([docs[i]])
                    cosine_similarities = cosine_similarity(tfidf_query, tfidf_corpus, dense_output=False).toarray().flatten()
                    # position = np.argmax(cosine_similarities)
                    sorted_positions = np.argsort(cosine_similarities)[::-1]
                    position = sorted_positions[0] if len(sorted_positions) > 1 else -1
                    description = ""
                    commit_messages = ""
                    linked_issue_titles = ""
                    desc_content = re.findall(r'<desc>(.*?)</desc>', sources_text[position], re.DOTALL)
                    cmt_content = re.findall(r'<cmt>(.*?)</cmt>', sources_text[position], re.DOTALL)
                    iss_content = re.findall(r'<iss>(.*?)</iss>', sources_text[position], re.DOTALL)
                    for desc in desc_content:
                        description += desc
                    for cmt in cmt_content:
                        commit_messages += cmt
                    for iss in iss_content:
                        linked_issue_titles += iss

                    pre_data = {
                        "description": description,
                        "commit_messages": commit_messages,
                        "linked_issue_titles": linked_issue_titles,
                        "title": sources_title[position]
                    }
                    file.write(pre_data)
            logging.info("Finish Matching Data!")

        else:
            logging.info("No Matching Way!")
