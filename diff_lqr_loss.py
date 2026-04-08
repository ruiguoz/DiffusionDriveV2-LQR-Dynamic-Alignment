import torch
import torch.nn.functional as F


def ultimate_zero_loop_lqr_loss(
    pred_traj_xyh: torch.Tensor,        # [B, T, 3] 模型预测轨迹
    initial_state: torch.Tensor,        # [B, 5]   (x, y, yaw, vx, steer)
    ref_vel_horizon: torch.Tensor,      # [B, T, H] 参考速度 H=10
    ref_curv_horizon: torch.Tensor,     # [B, T, H] 参考曲率 H=10
    dt: float = 0.1,
    wheelbase: float = 3.089,
    tracking_horizon: int = 10,
    q_lon: float = 10.0,
    r_lon: float = 1.0,
    q_lat: list = [1.0, 10.0, 0.0],
    r_lat: float = 1.0,
    stop_vel: float = 0.2,
    stop_gain: float = 0.5,
    max_steer: float = torch.pi / 3.0,
):
    B, T, _ = pred_traj_xyh.shape
    H = tracking_horizon
    device = pred_traj_xyh.device
    dtype = pred_traj_xyh.dtype

    # 参考轨迹
    rx = pred_traj_xyh[..., 0]
    ry = pred_traj_xyh[..., 1]
    ryaw = pred_traj_xyh[..., 2]

    # 初始状态
    x0 = initial_state[:, [0]]
    y0 = initial_state[:, [1]]
    yaw0 = initial_state[:, [2]]
    v0 = initial_state[:, [3]]
    steer0 = initial_state[:, [4]]

    # ======================== 固定矩阵 ========================
    Q = torch.diag(torch.tensor(q_lat, device=device, dtype=dtype))
    R = torch.tensor(r_lat, device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)
    B_lon = H * dt
    inv_lon = -1.0 / (B_lon ** 2 * q_lon + r_lon)

    # ======================== H 步矩阵 A 累积（无循环）=========================
    v_h = ref_vel_horizon
    k_h = ref_curv_horizon

    A_k = I3.view(1, 1, 1, 3, 3).repeat(B, T, H, 1, 1)
    A_k[..., 0, 1] = v_h * dt
    A_k[..., 1, 2] = v_h * dt / wheelbase

    A = A_k[:, :, 0]
    for i in range(1, H):
        A = torch.einsum('...ij,...jk->...ik', A_k[:, :, i], A)

    B = torch.zeros(B, T, 3, 1, device=device, dtype=dtype)
    B[..., 2, 0] = dt
    g = torch.zeros(B, T, 3, device=device, dtype=dtype)

    # ======================== 横向误差 ========================
    dx = x0 - rx
    dy = y0 - ry
    c = torch.cos(ryaw)
    s = torch.sin(ryaw)
    e_lat = -dx * s + dy * c
    e_head = torch.atan2(torch.sin(yaw0 - ryaw), torch.cos(yaw0 - ryaw))
    e_steer = steer0.expand(B, T)
    X = torch.stack([e_lat, e_head, e_steer], dim=-1)

    # ======================== 纵向 LQR ========================
    v_ref_T = ref_vel_horizon[..., -1]
    stop_mask = (v0 <= stop_vel) & (v_ref_T <= stop_vel)
    err_v = v0 - v_ref_T
    accel = inv_lon * B_lon * q_lon * err_v
    accel = torch.where(stop_mask, -stop_gain * err_v, accel)

    # ======================== 横向 LQR ========================
    AX = torch.einsum('btij,btj->bti', A, X)
    err0 = AX + g
    err0[..., 1] = torch.atan2(torch.sin(err0[..., 1]), torch.cos(err0[..., 1]))

    BT = B.transpose(-2, -1)
    BTQ = torch.einsum('btij,jk->btik', BT, Q)
    S = torch.einsum('btij,btjk->btik', BTQ, B).squeeze() + R
    inv_S = -1.0 / S
    tail = torch.einsum('btij,btj->bti', BTQ, err0).squeeze(-1)
    steer_rate = inv_S * tail
    steer_rate = torch.where(stop_mask, torch.zeros_like(steer_rate), steer_rate)

    # ======================== 自行车模型（全向量化）=========================
    v_seq = v0 + torch.cumsum(accel * dt, dim=1)
    steer_seq = steer0 + torch.cumsum(steer_rate.unsqueeze(-1) * dt, dim=1)
    steer_seq = steer_seq.clamp(-max_steer, max_steer)

    cy, sy = torch.cos(yaw0), torch.sin(yaw0)
    dx_ego = v_seq * cy * dt
    dy_ego = v_seq * sy * dt
    dyaw_ego = (v_seq / wheelbase) * torch.tan(steer_seq) * dt

    x_traj = x0 + torch.cumsum(dx_ego, dim=1)
    y_traj = y0 + torch.cumsum(dy_ego, dim=1)
    yaw_traj = yaw0 + torch.cumsum(dyaw_ego, dim=1)
    yaw_traj = torch.atan2(torch.sin(yaw_traj), torch.cos(yaw_traj))

    # ======================== Loss ========================
    track_xy = torch.stack([x_traj, y_traj], dim=-1)
    loss = F.mse_loss(track_xy, pred_traj_xyh[..., :2])
    return loss