import logging
import backoff
import jsonlines
import requests
from openai import OpenAI
import json
from tqdm import tqdm
import transformers
import torch

class BaseLLM:

    def __init__(self, model_id: str, generation_kwargs):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs
    def get_chat_completion(self, prompt: str) -> str:
        return None

    def get_completion_file(self, input_path: str, output_path: str, limit_test: int) -> None:
        with jsonlines.open(input_path, "r") as reader:
            for i, line in tqdm(enumerate(reader), "Generating predictions"):

                with jsonlines.open(output_path, "a") as writer:
                    if i == 4023:
                        writer.write(
                            {"Prediction": None, "Title": line["title"]}
                        )
                    else:
                        try:
                            writer.write(
                            {"Prediction": self.get_chat_completion(line["prompt"]), "Title": line["title"]}
                            )
                        except Exception:
                            logging.exception(f"Encountered API error for example {i}")
                            writer.write({"Prediction": None, "Target": line["target"]})
                if limit_test > 0 and i > limit_test:
                    break



class DeepSeekerV2Utils(BaseLLM):

    def __init__(self, model_id: str, generation_kwargs):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs

    @backoff.on_exception(backoff.constant, Exception, interval=10)
    def get_chat_completion(self, prompt: str) -> str:
        #add your api_key
        client = OpenAI(api_key="********", base_url="https://api.deepseek.com/")
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant that creates Pull Request Title for given information."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            **self._generation_kwargs
        )
        return response.choices[0].message.content



class HuggingFaceLLM(BaseLLM):

    def __init__(self, model_id: str, device: int, generation_kwargs):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs
        model_path = {
                        "llama3-8b" : "meta-llama/Meta-Llama-3-8B-Instruct",
                        "codellama-7b" : "codellama/CodeLlama-7b-Instruct-hf",
                        "deepseeker-6.7b" : "deepseek-ai/deepseek-coder-6.7b-instruct",
                        "blue" : "vivo-ai/BlueLM-7B-Base-32K"
                     }
        assert model_path[model_id] != None, "Model is not supported"
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=model_path[model_id],
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            token='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'
        )

    @backoff.on_exception(backoff.constant, Exception, interval=10)
    def get_chat_completion(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant that write one Pull Request Title for some given information.Only output one of the most possible Pull Request Title.Just give me result,don't say anything else like 'Here is the Pull Request Title:'"},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self._pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self._pipeline(
            chat_prompt,
            max_new_tokens=self._generation_kwargs.max_tokens,
            #eos_token_id=terminators,
            do_sample=True,
            temperature=self._generation_kwargs['temperature'],
            top_p=self._generation_kwargs['top_p'],
        )
        result = outputs[0]["generated_text"][len(chat_prompt):]
        return result
    

# GPT3.5
class OpenAIUtils(BaseLLM):
    def __init__(self, model_id: str, generation_kwargs: dict):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs
        self.client = OpenAI(api_key="*******",
                             base_url="https://api.chatgptid.net/v1")


    @backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=1)
    def get_chat_completion(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful programming assistant that creates Pull Request Title for given information."},
                    {"role": "user", "content": prompt},
                ],
                **self._generation_kwargs
            )

        except Exception:
            return None
        # print(response)
        return response.choices[0].message.content


#Llama3-70B
class LlamaUtils(BaseLLM):

    def __init__(self, model_id: str, generation_kwargs):
        self._model_id = model_id
        self._generation_kwargs = generation_kwargs
        self._access_token = self.get_access_token()

    @staticmethod
    def get_access_token():
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=******&client_secret=*******"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    @backoff.on_exception(backoff.constant, Exception, interval=10)
    # @backoff.on_exception(backoff.expo, Exception)
    def get_chat_completion(self, prompt: str) -> str:
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_70b?access_token=" + self._access_token
            payload = json.dumps({
                "prompt": prompt,
                "temperature": self._generation_kwargs['temperature'],
                "top_p": self._generation_kwargs['top_p'],
            })
            payload = json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},
                    # {"role": "assistant",
                    #  "content": "You are a helpful programming assistant that creates Pull Request Title for given information."},

                ],
                "temperature": self._generation_kwargs['temperature'],
                "top_p": self._generation_kwargs['top_p'],
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)


            result = ""
            if response.text != None and json.loads(response.text)['result'] != None:
                result = json.loads(response.text)['result']
            logging.info(result)
        except Exception as e:
            return None
        return result
