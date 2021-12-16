from tfcv.detection.arguments import PARSER
from tfcv.detection.config import CONFIG
from tfcv.detection.dataset.dataset import Dataset

from argparse import Namespace

if __name__ == '__main__':
    # setup params
    arguments = PARSER.parse_args()
    params = Namespace(**{**vars(CONFIG), **vars(arguments)})

    dataset = Dataset(params)