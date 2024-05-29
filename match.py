import os
import sys

from source.process import MatchingCaculator
import hydra
from omegaconf import DictConfig, OmegaConf
from source.utils import timer


@timer(timer_name="prompting")
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    source_no_test = os.path.join(cfg.dataset.root_dir, cfg.dataset.source_no_test)
    source_train = os.path.join(cfg.dataset.root_dir, cfg.dataset.source_train)
    source_no_train = os.path.join(cfg.dataset.root_dir, cfg.dataset.source_no_train)
    output_path = os.path.join(cfg.dataset.root_dir, f"{cfg.matching}/test_{cfg.dataset.input_path}")
    if sys.platform.startswith("win"):
        output_path = hydra.utils.to_absolute_path(output_path)
        source_no_test = hydra.utils.to_absolute_path(source_no_test)
        source_no_train = hydra.utils.to_absolute_path(source_no_train)
        source_train = hydra.utils.to_absolute_path(source_train)
    print(output_path)
    match = MatchingCaculator(
        prompt_configuration=cfg.dataset.prompt_configuration,
    )
    #开始匹配最相似的text，通过no-token匹配,写入的是带有token的，到jsonl文件
    match.match_data(source_no_test, source_no_train, source_train, cfg.matching, output_path, cfg.dataset.use_cache)


if __name__ == "__main__":
    main()
