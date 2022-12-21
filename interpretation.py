from tqdm import tqdm
from typing import Generic, List, Type, TypeVar

from dig.xgraph.method import *
import numpy as np
import torch
import yaml

from .main import Processor, init_seed, get_parser

T = TypeVar('T')
class InterpretationProcessor(Processor, Generic[T]):
    def __init__(self, arg):
        super().__init__(arg)

        self._xmethod: T = T(self.model, explain_graph=True)
    
    def interpret(self, loader_name=['test']):
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            for ln in loader_name:

                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    
                    out_x = self._xmethod(data.x, data.edge_index,
                        sparsity=0.0,
                        num_classes=60)
                    edge_masks = out_x[0]
                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    y_pred = output.argmax(-1)
                    edge_mask = edge_masks[y_pred.item()].data

        # Empty cache after evaluation
        torch.cuda.empty_cache()


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = InterpretationProcessor[GradCAM](arg)
    processor.interpret()


if __name__ == '__main__':
    main()