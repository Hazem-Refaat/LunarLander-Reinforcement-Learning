# Reinforcement Learning with Stable Baselines3 and OpenAI Gym

This project demonstrates how to train and test a Reinforcement Learning (RL) model using Stable Baselines3 and OpenAI Gym. The RL agent will be trained on the LunarLander-v2 environment provided by OpenAI Gym, and then evaluated on the trained model's performance.

## Prerequisites

Make sure you have the following dependencies installed:

- gym
- stable_baselines3
- numpy

## Usage

### Training the PPO Agent

To train the Proximal Policy Optimization (PPO) agent, you can run the `train_ppo()` function. This function will create the Lunar Lander environment, initialize the PPO agent, train it for 500,000 timesteps, and save the trained model.

```python
from stable_baselines3 import PPO

def train_ppo():
    # Create the Lunar Lander environment
    env = gym.make('LunarLander-v2')

    # Create the PPO agent
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=500000)

    # Save the trained model
    model.save('ppo_lunar_lander500k')

    # Close the environment
    env.close()

if __name__ == "__main__":
    train_ppo()
```

### Testing the Trained Model

After training the PPO agent, you can evaluate its performance using the `test_model()` function. This function loads the trained model, runs it on the Lunar Lander environment for a few episodes, and prints the total reward obtained in each episode.

```python
def test_model():
    # Create the Lunar Lander environment
    env = gym.make('LunarLander-v2')

    # Load the trained PPO agent
    model = PPO.load('ppo_lunar_lander500k')

    # Evaluate the agent
    eval_episodes = 5
    for episode in range(eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Render the environment
            env.render()

            # Get the action from the model
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, _ = env.step(action)

            # Accumulate the total reward
            total_reward += reward

        # Print the reward for each episode
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    # Close the environment
    env.close()

# Call the test_model function
test_model()
```

## Understanding The Environment

To know how many actions the Lunar Lander environment has, you can use the `env.action_space` attribute. For example:

```python
print(env.action_space)
```

## Note

This project uses Proximal Policy Optimization (PPO) as the RL algorithm and trains the agent for 500,000 timesteps. You can modify the hyperparameters and the number of timesteps to explore different settings and improve the agent's performance.

Ensure that you have a compatible GPU and appropriate drivers installed if you want to leverage GPU acceleration for faster training.

And i will include a 100,000 timesteps model and a 500,000 timesteps model