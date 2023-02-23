from mmcv.utils import Registry

VOXEL_ENCODERS = Registry('voxel_encoder')
MIDDLE_ENCODERS = Registry('middle_encoder')
FUSION_LAYERS = Registry('fusion_layer')

# ACTIVATION_LAYERS = Registry('activation layer')
DROPOUT_LAYERS = Registry('drop out layers')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')
