class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict()
config.MODEL = AttrDict()
config.MODEL.EXTRA = {
    'STAGE2': {
        'NUM_MODULES': 1,
        'NUM_BRANCHES': 2,
        'NUM_BLOCKS': [4, 4],
        'NUM_CHANNELS': [18, 36],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE3': {
        'NUM_MODULES': 4,
        'NUM_BRANCHES': 3,
        'NUM_BLOCKS': [4, 4, 4],
        'NUM_CHANNELS': [18, 36, 72],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE4': {
        'NUM_MODULES': 3,
        'NUM_BRANCHES': 4,
        'NUM_BLOCKS': [4, 4, 4, 4],
        'NUM_CHANNELS': [18, 36, 72, 144],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'FINAL_CONV_KERNEL': 1
}
config.MODEL.NUM_JOINTS = 106  # Số landmark
config.MODEL.PRETRAINED = ''  # Nếu có checkpoint pretrained
config.MODEL.INIT_WEIGHTS = False  # True nếu muốn load pretrained
