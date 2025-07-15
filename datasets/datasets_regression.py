import sys
sys.path.append('..')

from utils.utils_pfld import calculate_pitch_yaw_roll


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class LandmarkDataset(Dataset):
    def __init__(self, anno_file, image_size=112, is_train=True, repeat=10, mirror_file=None):
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        self.lines = lines
        assert(len(lines) == 207)
        self.landmark = np.asarray(list(map(float, lines[:196])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, lines[196:200])),dtype=np.int32)
        flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        

        self.image_size = image_size    
        self.is_train = is_train
        self.repeat = repeat
        self.mirror_idx = None
        if mirror_file:
            with open(mirror_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                self.mirror_idx = list(map(int, lines[0].strip().split(',')))
        

    def __len__(self):
        return len(self.lines) * (self.repeat if self.is_train else 1)