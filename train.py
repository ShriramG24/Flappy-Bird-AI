from model import LinearDQN, ReplayMemory, Agent, Experience, QValues
from gym_env import FlappyBirdEnv
from torch.nn import functional as F
import torch

DISCOUNT = 0.999
BATCH_SIZE = 256
E_START = 1
E_END = 0.01
E_DECAY = 0.01
TNET_UPDATE = 10
LR = 1e-5
NUM_EPS = 1000
MEMORY_SIZE = 100000

env = FlappyBirdEnv()
memory = ReplayMemory(MEMORY_SIZE)
agent = Agent(env.action_space.n, E_START, E_END, E_DECAY)
policy = LinearDQN(env.display.width, env.display.height).to("cpu")
target = LinearDQN(env.display.width, env.display.height).to("cpu")

target.load_state_dict(policy.state_dict())
target.eval()

optimizer = torch.optim.Adam(params=policy.parameters(), lr=LR)


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


for episode in range(NUM_EPS):
    obs = env.reset()
    for timestep in range(100):
        env.render()
        action = agent.select_action(obs, policy)
        obs, reward, done, info = env.step(action)
        next_obs = env.process_image()
        memory.push(Experience(obs, action, next_obs, reward))
        obs = next_obs
        if memory.can_sample(BATCH_SIZE):
            obss, actions, rewards, next_obss = extract_tensors(memory.sample(BATCH_SIZE))
            current_q_values = QValues.get_current(policy, obss, actions)
            next_q_values = QValues.get_next(target, next_obss)
            target_q_values = (next_q_values * DISCOUNT) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            print("Episode finished after {} timesteps.".format(timestep + 1))
            break
        if episode % TNET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

env.close()


