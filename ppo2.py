import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import Counter, deque
import random
from task_allocation1 import *


class ActorCritic(nn.Module):
    def __init__(self, state_dim, max_actions=50):
        super(ActorCritic, self).__init__()
        self.max_actions = max_actions

        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 修改点1: 使用单一演员网络输出所有无人机类型的分数
        self.actor = nn.Linear(256, 3)  # 输出三种无人机类型的分数

        # 评论家网络保持不变
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        features = self.shared_layers(state)

        # 修改点2: 单一网络输出所有分数
        scores = self.actor(features)  # [batch_size, 3]

        # 分解为三种无人机的分数
        interceptor_score = scores[:, 0].unsqueeze(1)  # 保持二维形状 [batch_size, 1]
        recon_score = scores[:, 1].unsqueeze(1)
        escort_score = scores[:, 2].unsqueeze(1)

        # 评论家网络输出状态价值
        state_value = self.critic(features)

        return (interceptor_score, recon_score, escort_score), state_value


class PPOAgent:
    def __init__(self, state_dim, max_actions=50, lr=3e-4, gamma=0.99,
                 epsilon=0.2, entropy_coef=0.01, batch_size=64,
                 update_epochs=10, clip_param=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.max_actions = max_actions

        # 使用新的网络架构
        self.policy = ActorCritic(state_dim, max_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 其他参数保持不变...
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.clip_param = clip_param
        self.memory = deque(maxlen=2048)

        # 添加训练统计
        self.total_timesteps = 0
        self.total_episodes = 0
        self.best_eval_score = -float('inf')

    def get_action(self, state, available_counts):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 修改点3: 现在使用单一网络输出的分数
            (interceptor_score, recon_score, escort_score), _ = self.policy(state_tensor)

        # 根据可用数量生成概率分布
        actions = []
        action_probs = []

        # 为每种无人机类型生成动作概率
        scores = [interceptor_score.squeeze(0), recon_score.squeeze(0), escort_score.squeeze(0)]

        for i, (score, max_count) in enumerate(zip(scores, available_counts)):
            # 限制最大可选数量不超过可用数量和预设最大值
            max_count = min(max_count, self.max_actions)
            if max_count <= 0:  # 如果没有可用的无人机
                actions.append(0)
                action_probs.append(1.0)  # 概率为1选择0
                continue

            # 生成动作分数 (0到max_count)
            action_logits = torch.zeros(max_count + 1).to(self.device)

            # 使用分数生成概率分布
            for action_val in range(max_count + 1):
                # 计算动作值对应的分数 (线性关系)
                action_logits[action_val] = score * (action_val / max_count)

            # 转换为概率分布
            action_probs_i = F.softmax(action_logits, dim=-1)

            # 采样动作
            action_i = np.random.choice(range(max_count + 1), p=action_probs_i.cpu().numpy())

            actions.append(action_i)
            action_probs.append(action_probs_i[action_i].item())

        return (actions[0], actions[1], actions[2]), action_probs

    def remember(self, state, action, action_probs, reward, next_state, done, available_counts):
        self.memory.append((state, action, action_probs, reward, next_state, done, available_counts))
        self.total_timesteps += 1  # 跟踪总时间步数

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, old_probs, rewards, next_states, dones, available_counts = zip(*minibatch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)  # (batch, 3)
        old_probs = torch.FloatTensor(np.array(old_probs)).to(self.device)  # (batch, 3)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算折扣回报
        returns = self.compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算优势函数
        with torch.no_grad():
            _, values = self.policy(states)
            advantages = returns - values.squeeze()

        # 多轮更新
        for _ in range(self.update_epochs):
            # 获取新策略分数
            (interceptor_score, recon_score, escort_score), values = self.policy(states)

            # 计算新策略概率
            new_probs = torch.zeros_like(old_probs)

            for i, (score, counts) in enumerate(zip(
                    [interceptor_score, recon_score, escort_score],
                    zip(*available_counts)  # 转置可用计数
            )):
                # 为批次中的每个样本计算新概率
                batch_probs = []
                for j in range(self.batch_size):
                    max_count = min(counts[j], self.max_actions)
                    action_val = actions[j, i]

                    if max_count <= 0:
                        # 如果没有可用无人机，概率为1选择0
                        batch_probs.append(torch.tensor(1.0).to(self.device))
                        continue

                    # 生成动作分数
                    action_logits = torch.zeros(max_count + 1).to(self.device)
                    for a in range(max_count + 1):
                        action_logits[a] = score[j] * (a / max_count)

                    # 计算选定动作的概率
                    action_probs_i = F.softmax(action_logits, dim=-1)
                    batch_probs.append(action_probs_i[action_val])

                if not batch_probs:  # 如果列表为空
                    continue
                new_probs[:, i] = torch.stack(batch_probs)

            # 计算比率（三种动作的联合概率）
            ratios = (new_probs / old_probs).prod(dim=1)  # 计算每个样本的联合概率比

            # 计算裁剪后的目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算评论家损失
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # 计算熵正则化
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).mean()

            # 总损失
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空记忆
        self.memory.clear()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))


def generate_enemy_formation(state):
    # 从状态中获取敌方剩余无人机数量
    enemy_remaining = state.get('current_enemy_formation_remaining', {})

    # 获取剩余数量，如果不存在则使用默认值
    ground_attack_remaining = enemy_remaining.get('ground_attack', 10)
    recon_remaining = enemy_remaining.get('recon', 10)
    escort_remaining = enemy_remaining.get('escort', 30)

    # 确保生成的数量不超过剩余数量
    ground_attack = min(np.random.randint(1, 5), ground_attack_remaining)
    recon = min(np.random.randint(1, 3), recon_remaining)
    escort = min(np.random.randint(0, 6), escort_remaining)

    # 确保满足最小约束（至少1个侦察机和1个对地攻击机）
    ground_attack = max(ground_attack, 1) if ground_attack_remaining > 0 else 0
    recon = max(recon, 1) if recon_remaining > 0 else 0

    return {
        'ground_attack': ground_attack,
        'recon': recon,
        'escort': escort
    }


def generate_battlefield_coords():
    # 假设SCREEN_WIDTH和SCREEN_HEIGHT已在环境中定义
    return (
        np.random.uniform(SCREEN_WIDTH * 0.3, SCREEN_WIDTH * 0.7),
        np.random.uniform(SCREEN_HEIGHT * 0.3, SCREEN_HEIGHT * 0.7)
    )


def train_ppo(env, agent, episodes=1000, max_steps=100,
              eval_interval=1000, eval_episodes=5, save_dir="results"):
    """增强的训练函数，包含定期评估"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 初始化评估结果文件
    eval_file = os.path.join(save_dir, "evaluation_results.csv")
    with open(eval_file, "w") as f:
        f.write("episode,timestep,avg_reward,std_reward,win_rate,model_checkpoint\n")

    print("Starting PPO training with periodic evaluation...")
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        # 使用当前状态生成敌人编队
        enemy_formation = generate_enemy_formation(state)
        battlefield_coords = generate_battlefield_coords()

        while not done and step < max_steps:
            # 构建状态向量
            state_vector = [
                enemy_formation['ground_attack'],
                enemy_formation['recon'],
                enemy_formation['escort'],
                state['friendly_remaining']['interceptor'],
                state['friendly_remaining']['recon'],
                state['friendly_remaining']['escort']
            ]

            # 获取可用无人机数量
            available_counts = [
                state['friendly_remaining']['interceptor'],
                state['friendly_remaining']['recon'],
                state['friendly_remaining']['escort']
            ]

            # 获取动作（部署数量） - 使用可用数量信息
            (interceptor_deploy, recon_deploy, escort_deploy), action_probs = agent.get_action(
                state_vector, available_counts
            )

            # 构建动作字典
            action_dict = {
                'interceptor': interceptor_deploy,
                'recon': recon_deploy,
                'escort': escort_deploy
            }

            # 执行环境步骤
            next_state, reward, done, info = env.step(
                action=action_dict,
                enemy_formation=enemy_formation,
                battlefield_coords=battlefield_coords
            )

            # 构建下一个状态向量
            next_state_vector = [
                enemy_formation['ground_attack'],
                enemy_formation['recon'],
                enemy_formation['escort'],
                next_state['friendly_remaining']['interceptor'],
                next_state['friendly_remaining']['recon'],
                next_state['friendly_remaining']['escort']
            ]

            # 下一个状态的可用数量
            next_available_counts = [
                next_state['friendly_remaining']['interceptor'],
                next_state['friendly_remaining']['recon'],
                next_state['friendly_remaining']['escort']
            ]

            # 存储经验（添加可用数量信息）
            agent.remember(
                state_vector,
                [interceptor_deploy, recon_deploy, escort_deploy],
                action_probs,
                reward,
                next_state_vector,
                done,
                available_counts  # 添加当前可用数量
            )

            # 更新状态
            state = next_state
            total_reward += reward
            step += 1

            # 更新PPO策略
            agent.update()

            # 为下一步生成新的敌人编队和战场坐标
            enemy_formation = generate_enemy_formation(state)
            battlefield_coords = generate_battlefield_coords()

        episode_rewards.append(total_reward)

        # 定期评估
        if episode % eval_interval == 0 and episode > 0:
            # 创建新的测试环境（禁用渲染）
            test_env = UAVCombatEnv(
                initial_red_uav_counts=Counter({'interceptor': 40, 'recon': 10, 'escort': 30}),
                initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
                render_mode=False
            )

            # 执行评估
            eval_results = evaluate_agent(agent, test_env, eval_episodes, max_steps)

            # # 保存评估结果
            checkpoint_name = f"ppo_model_ep{episode}.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            # agent.save(checkpoint_path)

            # 记录到CSV文件
            with open(eval_file, "a") as f:
                f.write(f"{episode},{eval_results['timestep']},{eval_results['avg_reward']:.2f},")
                f.write(f"{eval_results['std_reward']:.2f},{eval_results['win_rate']:.1f},{checkpoint_path}\n")

            # 保存最佳模型
            if eval_results["avg_reward"] > agent.best_eval_score:
                agent.best_eval_score = eval_results["avg_reward"]
                best_model_path = os.path.join(save_dir, "ppo_best_model.pth")
                agent.save(best_model_path)
                print(f"New best model saved with score {agent.best_eval_score:.2f}")

            test_env.close()

        # 保存最终模型
    final_model_path = os.path.join(save_dir, "ppo_final_model.pth")
    agent.save(final_model_path)
    print("Training completed!")
    return episode_rewards


def evaluate_agent(agent, env, eval_episodes=5, max_steps=100):
    """在测试环境中评估智能体性能"""
    print(f"\nStarting evaluation with {eval_episodes} episodes...")
    total_rewards = []
    win_rates = []

    for ep in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # 使用当前状态生成敌人编队
        enemy_formation = generate_enemy_formation(state)
        battlefield_coords = generate_battlefield_coords()

        while not done and step < max_steps:
            # 构建状态向量
            state_vector = [
                enemy_formation['ground_attack'],
                enemy_formation['recon'],
                enemy_formation['escort'],
                state['friendly_remaining']['interceptor'],
                state['friendly_remaining']['recon'],
                state['friendly_remaining']['escort']
            ]

            # 获取可用无人机数量
            available_counts = [
                state['friendly_remaining']['interceptor'],
                state['friendly_remaining']['recon'],
                state['friendly_remaining']['escort']
            ]

            # 使用训练好的策略获取动作（不探索）
            with torch.no_grad():
                actions, _ = agent.get_action(state_vector, available_counts)

            # 构建动作字典
            action_dict = {
                'interceptor': actions[0],
                'recon': actions[1],
                'escort': actions[2]
            }

            # 执行环境步骤
            next_state, reward, done, info = env.step(
                action=action_dict,
                enemy_formation=enemy_formation,
                battlefield_coords=battlefield_coords
            )

            # 更新状态
            state = next_state
            episode_reward += reward
            step += 1

            # 为下一步生成新的敌人编队和战场坐标
            enemy_formation = generate_enemy_formation(state)
            battlefield_coords = generate_battlefield_coords()

        total_rewards.append(episode_reward)
        win_rates.append(1 if info.get('win', False) else 0)  # 根据实际环境信息确定胜利

        print(f"Evaluation episode {ep + 1}/{eval_episodes}: Reward={episode_reward:.2f}")

    # 计算评估指标
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    win_rate = np.mean(win_rates) * 100

    print(f"Evaluation completed: Avg Reward={avg_reward:.2f} ± {std_reward:.2f}, Win Rate={win_rate:.1f}%")
    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "win_rate": win_rate,
        "timestep": agent.total_timesteps,
        "episode": agent.total_episodes
    }
# 主函数
if __name__ == "__main__":


    # 假设SCREEN_WIDTH和SCREEN_HEIGHT已在环境中定义
    global SCREEN_WIDTH, SCREEN_HEIGHT
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800

    # 初始化环境
    env = UAVCombatEnv(
        initial_red_uav_counts=Counter({'interceptor': 40, 'recon': 10, 'escort': 30}),
        initial_blue_uav_counts=Counter({'ground_attack': 10, 'recon': 10, 'escort': 30}),
        render_mode=False  # 训练时禁用渲染
    )

    # 定义状态维度和最大动作数
    state_dim = 6  # 新MDP建模：敌方3种 + 我方3种无人机数量
    max_actions = 50  # 最大可选动作数

    # 创建PPO智能体
    agent = PPOAgent(state_dim, max_actions, lr=1e-4)

    # 开始训练（增加评估参数）
    rewards = train_ppo(
        env,
        agent,
        episodes=2000000,
        max_steps=100,
        # render_every=500000,
        eval_interval=5000,  # 每5000轮评估一次
        eval_episodes=500,  # 每次评估500个回合
        save_dir="training_results"  # 结果保存目录
    )

    # 关闭环境
    env.close()