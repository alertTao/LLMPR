import os

#add your cache_home
os.environ["XDG_CACHE_HOME"] = "***/***"
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from source.process import DataPreprocessor, StringPreprocessor, Metric, DataPreprocessorMatching, \
    DataPreprocessorMatchingShot
from source.llm import BlueLLM, HuggingFaceLLM, DeepSeekerV2Utils
from source.utils import timer
import sys
import wandb

@timer(timer_name="prompting")
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    # 1. load and process data
    preprocessor = DataPreprocessor(
        prompt_configuration=cfg.dataset.prompt_configuration,
    )

    preprocessor_matching = DataPreprocessorMatching(
        prompt_configuration=cfg.dataset.prompt_configuration,
    )

    preprocessor_matching_shot = DataPreprocessorMatchingShot(
        prompt_configuration=cfg.dataset.prompt_configuration,
    )

    # source_test = os.path.join(cfg.dataset.root_dir, cfg.dataset.source_test)
    input_path = os.path.join(cfg.dataset.root_dir, cfg.dataset.input_path)
    processed_path = os.path.join(cfg.dataset.root_dir, "process",
                                  f"{cfg.dataset.prompt_configuration}_{cfg.dataset.input_path}")
    if sys.platform.startswith("win"):
        processed_path = hydra.utils.to_absolute_path(processed_path)
        input_path = hydra.utils.to_absolute_path(input_path)
    print(processed_path)

    # # first process
    # str_preprocessor = StringPreprocessor(
    #     prompt_configuration=cfg.dataset.prompt_configuration,
    # )

    # str_preprocessor.process_file(
    #     input_path=source_test,
    #     output_path=input_path,
    #     use_cache=cfg.dataset.use_cache,
    # )

    # 判断是否需要最相似示例
    if cfg.matching == 'BM25' or cfg.matching == 'TF-idf':
        if cfg.shot == 1:
            matching_path = os.path.join(cfg.dataset.root_dir,
                                         f"{cfg.matching}/first_{cfg.dataset.input_path}")
            preprocessor_matching.process_file(
                input_path=input_path,
                output_path=processed_path,
                matching_path=matching_path,
                use_cache=cfg.dataset.use_cache,
                limit_test=cfg.limit_test
            )
        elif cfg.shot in [2, 3, 4, 5]:
            matching_dir = os.path.join(cfg.dataset.root_dir,
                                        f"{cfg.matching}")
            preprocessor_matching_shot.process_file(
                input_path=input_path,
                output_path=processed_path,
                matching_dir=matching_dir,
                input_path_file=cfg.dataset.input_path,
                limit_test=cfg.limit_test,
                shot=cfg.shot,
                use_cache=cfg.dataset.use_cache,
            )
        else:
            logging.info("Shot Error!!!")
            sys.exit()
    else:
        preprocessor.process_file(
            input_path=input_path,
            output_path=processed_path,
            use_cache=cfg.dataset.use_cache,
            limit_test=cfg.limit_test
        )

    # 2. load LLM model
    if cfg.model_id == 'blue':
        llm_model = BlueLLM(cfg.model_id, cfg.generation)
    elif cfg.model_id == 'deepseekerv2':
        llm_model = DeepSeekerV2Utils(cfg.model_id, cfg.generation)
    else:
        llm_model = HuggingFaceLLM(cfg.model_id, cfg.device, cfg.generation)

    # 3. revoke model
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    predict_output_path = os.path.join(
        output_path,
        f"{cfg.model_id}_{cfg.dataset.prompt_configuration}_{cfg.matching}_{cfg.shot}_{cfg.dataset.input_path}"
    )

    llm_model.get_completion_file(processed_path, predict_output_path, limit_test=cfg.limit_test)

    # 4. compute metric
    metric_output_path = os.path.join(
        output_path,
        f"{cfg.model_id}_{cfg.dataset.prompt_configuration}_{cfg.generation.temperature}_{cfg.matching}_{cfg.shot}_metric_{cfg.dataset.input_path}"
    )
    print(metric_output_path)
    logging.info("Metrics: ")
    metric = Metric()
    metric.rouge_compete(predict_output_path, metric_output_path)


if __name__ == "__main__":
    main()
