"""最终锁定模型工厂；只构造模型，不训练。"""
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from common.config import FINAL_MLP, FINAL_SVC

def make_mlp(**overrides):
    return MLPClassifier(**(FINAL_MLP | overrides))

def make_models():
    return {"svc": SVC(**FINAL_SVC), "mlp": make_mlp()}
