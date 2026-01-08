import torch
from tqdm import tqdm
import wandb

def flow_loss(z, log_det_j, graph=False,time_series=False):
    if graph:
        return torch.mean(
                torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))
                - torch.sum(log_det_j, dim=(1,))
            )
        # return torch.mean(
        #             torch.sum(0.5 * torch.sum(z**2, dim=(2,)), dim=(1,))
        #             - torch.sum(log_det_j, dim=(1,)) 
        #         )/ (z.shape[1])
    else:
        return torch.mean(0.5 * torch.sum(z**2, dim=(1,)) - log_det_j) 
    
def flow_training(
    model,
    data_loader,
    flow_optimizer,
    epoch_num=500,
    model_dir=None,
    model_save_freq=100,
    data_th=None,
    writer=None,
):
    for m in tqdm(range(epoch_num), desc="Flow training"):
        cnt = 0
        lf = 0
        for data in data_loader:
            prev_obs, obs, prev_r_obs, r_obs, act, rwd, _, done = data

            flow_optimizer.zero_grad()
            z, log_det_j = model(
                prev_obs[:, :, :2].flatten(start_dim=1).to(model.device)
            )
            loss = flow_loss(z, log_det_j)
            lf += loss.data.item()
            loss.backward()
            flow_optimizer.step()

            cnt += 1

        if not model_dir is None:
            if (m + 1) % model_save_freq == 0:
                model.set_anomaly_threshold(data_th)
                model.save_model(model_dir + f"/model_{m + 1}.pth")

        if writer != None:
            writer.add_scalar("loss/flow", lf / cnt, m + 1)


def grevnet_training(
    model,
    data_loader,
    flow_optimizer,
    epoch_num=500,
    model_dir=None,
    model_save_freq=100,
    data_th=None,
    writer=None,
    use_wandb = False,
):
    for m in tqdm(range(epoch_num), desc="Flow training"):
        cnt = 0
        lf = 0
        outlier_list = []
        for data in data_loader:
            prev_obs, obs, prev_r_obs, r_obs, act, rwd, _, done = data
            flow_optimizer.zero_grad()
            z, log_det_j = model(
                prev_obs.to(model.device),
            )
            loss = flow_loss(z, log_det_j, graph=True)
            lf += loss.data.item()
            loss.backward()
            flow_optimizer.step()

            cnt += 1
            # model.set_anomaly_threshold(data_th)
        # switching_threshold = model.get_switching_score(data_th)
        # for data in data_loader:
        #     prev_obs, obs, prev_r_obs, r_obs, act, rwd, _, done = data
        #     outlier = model.get_switching_score(prev_obs.to(model.device))
        #     if outlier.data.item() > switching_threshold:
        #         outlier_list.append(True)
        #     else:
        #         outlier_list.append(False) 
        # true_ratio = sum(outlier_list) / len(outlier_list)
        # false_ratio = 1 - true_ratio

        # print(f"OOD_ratio: {true_ratio:.2%}")
        # print(f"ID_ratio: {false_ratio:.2%}")

        if not model_dir is None:
            if (m + 1) % model_save_freq == 0:
                model.set_switching_threshold(data_th)
                model.save_model(model_dir + f"/model_{m + 1}.pth")
                if use_wandb:
                    wandb.log({
                        "threshold": data_th
                    })

        if writer != None:
            writer.add_scalar("loss/flow", lf / cnt, m + 1)
            if use_wandb:
                wandb.log({
                    "loss": lf / cnt,
                    "epoch": m+1,
                })



def grevnet_task_training(
    model,
    tasks,
    flow_optimizer,
    epoch_num=500,
    model_dir=None,
    model_save_freq=100,
    data_th=None,
    writer=None,
    time_series = False,
    _wandb = False,
):

    for m in tqdm(range(epoch_num), desc="Flow training"):
        cnt = 0
        lf = 0
        losses = []
        # preb_obs_series = torch.zeros(obs_length, num_Ped, 2)
        for task in tasks:
            # inner_cnt = 0
            for data in task:
                prev_obs, obs, prev_r_obs, r_obs, act, rwd, _, done = data
                flow_optimizer.zero_grad()
                z, log_det_j = model(
                    prev_obs.to(model.device),
                )
                loss = flow_loss(z, log_det_j, graph=True)
                # loss = flow_loss_sum(z, log_det_j, graph=True)
                losses.append(loss)
                # inner_cnt += 1
                # if inner_cnt > 0:
                #     break
        average_loss = sum(losses) / len(losses)
        lf += loss.data.item()
        # loss.backward()
        average_loss.backward()
        flow_optimizer.step()

        cnt += 1
                # model.set_anomaly_threshold(data_th)
        if not model_dir is None:
            if (m + 1) % model_save_freq == 0:
                model.set_switching_threshold(data_th)
                model.save_model(model_dir + f"/model_{m + 1}.pth")
                if _wandb:
                    wandb.log({
                        "threshold": data_th
                    })

        if writer != None:
            writer.add_scalar("loss/flow", lf / cnt, m + 1)
            if _wandb:
                wandb.log({
                    "loss": lf / cnt,
                    "epoch": m+1,
                })