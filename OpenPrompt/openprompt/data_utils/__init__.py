from yacs.config import CfgNode
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .utils import InputExample, InputFeatures
from .data_sampler import FewShotSampler

from openprompt.utils.logging import logger

PROCESSORS = {
    **TC_PROCESSORS,
}


def load_dataset(config: CfgNode, return_class=True, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset

    processor = PROCESSORS[dataset_config.name.lower()]()

    train_dataset = None
    valid_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {dataset_config.path}.")
        try:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {dataset_config.path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(dataset_config.path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {dataset_config.path}.")
    # checking whether downloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset

