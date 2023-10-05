from easydict import EasyDict as edict

recognize_method  = "insightface"   # "insightface" or "facenet"

def config():
    conf = edict()
    recognize_method  = "insightface"
    if recognize_method == "insightface":
        conf.threshold = 1.1066

    elif recognize_method == "facenet":
        conf.threshold = 161.8
    else:
        return conf
    return conf

