import os
from tqdm import tqdm
from typing import List

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from dig.xgraph.method.gradcam import GraphLayerGradCam
import torch
import yaml

from main import Processor, init_seed, get_parser
from ntu_visualize import ntu_skeleton_bone_pairs as bones, actions

def rgb_to_hex(r, g, b):
  return '#%02x%02x%02x' % (r, g, b)

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

        self.upsample = torch.nn.Upsample(scale_factor=(4, 1, 1), mode='nearest')

    def _get_node_weight(self, x, ex_label):
        attr = self.explain_method.attribute(x, ex_label).detach().relu()
        attr -= attr.amin(dim=(1, 2, 3), keepdim=True)
        attr /= attr.amax(dim=(1, 2, 3), keepdim=True)

        return attr

    def interpret(self, loader_name='test'):
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()

            l = len(self._indices_to_be_interpreted)
            for start in tqdm(range(0, l, self.arg.test_batch_size)):
                end = start + self.arg.test_batch_size
                data, label, _ = self.data_loader[loader_name].dataset.__getitem__(self._indices_to_be_interpreted[start:end])
                names = [self.data_loader[loader_name].dataset.sample_name[i] for i in self._indices_to_be_interpreted[start:end]]
                data = torch.tensor(data)
                label = torch.tensor(label)
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                
                # get prediction of the model
                output = self.model(data)
                y_pred = output.argmax(-1)
                
                del output

                # get interpretation weights
                N, _, _, _, M = data.size()
                nodes_weight = self._get_node_weight(data, y_pred)                
                nodes_weight_per_body = nodes_weight.chunk(M)
                nodes_weight  = torch.stack(nodes_weight_per_body, dim=-1) # N 1 T/4 V M
                nodes_weight = self.upsample(nodes_weight).squeeze(1) # N T V M
                
                # Visualize samples
                self.visualize_samples(data.cpu().numpy(), names, label, y_pred, nodes_weight.cpu().numpy())

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def visualize_samples(self, data, names, labels, preds, node_weights):
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        
        def animate(skeleton):
            frame_index = skeleton_index[0]
            ax.clear()
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])

            for i, j in bones:
                joint_locs = skeleton[:,[i,j]]
                weight = (node_weight1[frame_index, i] + node_weight1[frame_index, j]) / 2
                r, g, b, _ = mapper.to_rgba(weight)
                color = rgb_to_hex(int(r * 255), int(g * 255), int(b * 255))
                
                # plot them
                ax.plot(joint_locs[0],joint_locs[1],joint_locs[2], color=color)
            
            plt.title('Skeleton {} Frame #{} of 300\n (Action: {}, Predicted: {})'.format(index, skeleton_index[0], action_name, predicted_name))
            skeleton_index[0] += 1
            skeleton_index[0] = min(skeleton_index[0], node_weight1.shape[0] - 1)

            return ax

        for index, (skeletons, name, action_class, pred, node_weight) in \
                    enumerate(zip(data, names, labels, preds, node_weights)):
            
            action_class += 1
            pred += 1
            
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_xlim([-1,1])
            ax.set_ylim([-1,1])
            ax.set_zlim([-1,1])

            # get data

            action_name = actions[action_class.item()]
            predicted_name = actions[pred.item()]
            print(f'Sample name: {name}\nAction: {action_name}, Predicted: {predicted_name}\n')   # (C,T,V,M)

            # Pick the first body to visualize
            skeleton1 = skeletons[..., 0]   # out (C,T,V)
            node_weight1 = node_weight[..., 0] # out (T,V)

            skeleton_index = [0]
            skeleton_frames = skeleton1.transpose(1,0,2)
            ani = FuncAnimation(fig, animate, skeleton_frames)
            
            # saving to m4 using ffmpeg writer
            writervideo = animation.FFMpegWriter(fps=60)
            save_dir = os.path.join(self.arg.work_dir, 'interpretation', action_name)
            os.makedirs(save_dir, exist_ok=True)
            ani.save(os.path.join(save_dir, f'{name}.mp4'), writer=writervideo)
            plt.close()

            print(f"@@@\nVisualization done for {name}\n@@@", flush=True)


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