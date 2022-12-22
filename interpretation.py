import os
from tqdm import tqdm
from typing import List

from dig.xgraph.method.gradcam import GraphLayerGradCam
from dig.xgraph.models.utils import normalize
import torch
import yaml

from main import Processor, init_seed, get_parser

class InterpretationProcessor(Processor):
    def __init__(self, arg):
        super().__init__(arg)

        self.explain_method = GraphLayerGradCam(self.model, self.model.tcn3)
        
        self._indices_to_be_interpreted: List[int] = list()
        with open(os.path.join(self.arg.work_dir, "wrong-samples.txt"), "r") as f:
            lines = f.read().split("\n")
            for l in lines:
                index = l.split(",")[0].replace("tensor(", "").replace(")", "")
                self._indices_to_be_interpreted.append(int(index))

    def _get_node_weight(self, x, ex_label):
        attr = self.explain_method.attribute(x, ex_label).detach()
        node_weight = normalize(attr.relu())
        return node_weight

    def interpret(self, loader_name='test'):
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()

            l = len(self._indices_to_be_interpreted)
            for start in tqdm(range(0, l, self.arg.test_batch_size)):
                end = start + self.arg.test_batch_size
                data, label, _ = self.data_loader[loader_name].dataset.__getitem__(self._indices_to_be_interpreted[start:end])
                data = torch.tensor(data)
                N, _, _, _, M = data.size()
                label = torch.tensor(label)
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                
                # get prediction of the model
                output = self.model(data)
                y_pred = output.argmax(-1)
                
                del output

                # get interpretation weights
                node_weight = self._get_node_weight(data, y_pred)
                node_weight_per_body = node_weight.chunk(M)
                node_weight = torch.stack(node_weight_per_body, dim=-1)

        # Empty cache after evaluation
        torch.cuda.empty_cache()


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    processor = InterpretationProcessor(arg)
    processor.interpret()


if __name__ == '__main__':
    main()