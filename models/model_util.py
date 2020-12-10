from .org_pixnerf import PixelNeRFNet
from .nerf import NeRF
from .pix_nerf import PixelNeRF
from .encoder import SpatialEncoder, ImageEncoder
from .mlp import ImplicitNet
from .resnetfc import ResnetFC


def make_model(conf, *args, **kwargs):
    """ Placeholder to allow more model types """
    model_type = conf.get_string("type", "nerf")  # single
    if model_type == "nerf":
        net = NeRF(*args, **kwargs)
    elif model_type == "pixelnerf":
        net = PixelNeRFNet(conf, *args, **kwargs)
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net


def make_mlp(conf, d_xyz, d_dir, d_latent=0, **kwargs):
    mlp_type = conf.get_string("type", "nerf_mlp")  # mlp | resnet
    if mlp_type == "nerf_mlp":
        net = NeRF(in_channels_xyz=d_xyz, in_channels_dir=d_dir)
        pass
    elif mlp_type == "resnet_mlp":
        net = ResnetFC.from_conf(conf, d_in=d_xyz + d_dir, d_latent=d_latent, **kwargs)
    elif mlp_type == "implicit_mlp":
        net = ImplicitNet.from_conf(conf, d_in=d_xyz + d_dir + d_latent, **kwargs)
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string("type", "spatial")  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net