# load configs from yaml
# load functions
import ipdb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from treePoints.model_manager import ModelManager
from treePoints.post_manager import PostprocessorManager

import os
import hydra


@hydra.main(config_path='config', config_name='hyperps')
def main(cfg):

    model_manager = ModelManager()

    model_manager.load_files(cfg)
    model_manager.download_weights(cfg)
    model_manager.load_model(cfg)
    model_manager.predict(cfg)

    post_manager = PostprocessorManager()
    post_manager.processor_run(cfg)







if __name__ == "__main__":
    main()
