from easydict import EasyDict as edict


def config():
    conf = edict()
    conf.method = "insightface"  # "insightface" or "facenet"
    if conf.method == "insightface":
        conf.threshold = 1.1066
    elif conf.method == "facenet":
        conf.threshold = 1.
    else:
        return conf
    return conf

