import torch

def calculate_odpr_weights(model, rb, weight_history, batch_size=100, epsilon=1e-37):
    model.eval()
    device = model.device
    current_size = len(rb)
    
    adv_list = []
    
    for l in range(0, current_size, batch_size):
        r = min(l + batch_size, current_size)
        
        indices = torch.arange(l, r)
        batch = rb[indices].to(device) 

        with torch.no_grad():
            Q1, Q2 = model.critic(
                (batch["humans_obs"], batch["robot_obs"]), 
                batch["action"].squeeze()
            )
            qw_ref = torch.min(Q1, Q2).reshape(-1, 1)

            action_gen, _, _ = model.actor.sample((batch["humans_obs"], batch["robot_obs"]))
            v_act1, v_act2 = model.critic(
                (batch["humans_obs"], batch["robot_obs"]), 
                action_gen.detach()
            )
            qw_gen = torch.min(v_act1, v_act2).reshape(-1, 1)
            adv = torch.clamp(qw_ref - qw_gen, min=0.0)
            adv_list.append(adv.cpu())

    adv = torch.cat(adv_list).to(torch.float64).flatten()
    padv = adv - torch.nan_to_num(torch.min(adv))

    current_weight = padv / (torch.sum(padv) + epsilon) * current_size
    new_weight = weight_history[:current_size] * current_weight
    
    new_weight = (new_weight / (torch.sum(new_weight) + epsilon)) * current_size
    
    model.train()
    return new_weight