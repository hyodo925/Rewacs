from collections import OrderedDict
import torch
class VirtualActorUpdater:
    def __init__(self):
        self.virtual_params = {}

    def step(self, model, grads, step_name, lr, from_params=None):
        if from_params is None:
            prev_params = OrderedDict(
                (name, param.data.clone()) for name, param in model.named_parameters()
            )
        else:
            prev_params = from_params

        new_params = OrderedDict()
        for (name, _), grad in zip(model.named_parameters(), grads):
            if grad is None:
                grad = torch.zeros_like(prev_params[name])
            if grad.shape != prev_params[name].shape:
                raise RuntimeError(f"Shape mismatch at {name}: param={prev_params[name].shape}, grad={grad.shape}")
            new_params[name] = prev_params[name] - lr * grad

        self.virtual_params[step_name] = new_params

    def get(self, step_name):
        return self.virtual_params[step_name]
    

# from collections import OrderedDict
# import torch

# class VirtualActorUpdater:
#     def __init__(self):
#         self.virtual_params = {}

#     def step(self, model, grads, step_name, lr, from_params=None, target_names=None):
#         """
#         Args:
#             model: オリジナルのモデル
#             grads: 計算された勾配のリスト
#             step_name: 保存用のキー名
#             lr: 学習率
#             from_params: ベースとなるパラメータ（Noneの場合はmodelから取得）
#             target_names: 更新対象とするパラメータ名の部分一致リスト (例: ['mean_linear', 'log_std_logits'])
#                          None の場合は全層を更新します。
#         """
#         if from_params is None:
#             # modelの現在のパラメータをベースにする
#             prev_params = OrderedDict(
#                 (name, param.data.clone()) for name, param in model.named_parameters()
#             )
#         else:
#             # 前のステップで作った仮想パラメータをベースにする
#             prev_params = from_params

#         new_params = OrderedDict()
        
#         # model.named_parameters() と grads (勾配リスト) を同期させてループ
#         for (name, _), grad in zip(model.named_parameters(), grads):
#             # 1. 更新対象かどうかの判定
#             is_target = True
#             if target_names is not None:
#                 # target_namesに含まれる文字列がnameに含まれているかチェック
#                 is_target = any(tn in name for tn in target_names)

#             # 2. パラメータの更新処理
#             if is_target and grad is not None:
#                 if grad.shape != prev_params[name].shape:
#                     raise RuntimeError(f"Shape mismatch at {name}")
#                 # 更新対象であれば勾配を引く
#                 new_params[name] = prev_params[name] - lr * grad
#             else:
#                 # 更新対象外、または勾配がない場合は前の値をそのまま維持
#                 new_params[name] = prev_params[name]

#         self.virtual_params[step_name] = new_params

#     def get(self, step_name):
#         return self.virtual_params[step_name]

class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)