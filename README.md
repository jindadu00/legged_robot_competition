# 强化学习训练GO2翻越多种地形
【基于Isaac Gym的四足机器狗强化学习控制翻越梅花桩】 https://www.bilibili.com/video/BV1qQ4VefEa6/?share_source=copy_web&vd_source=ac75510eaafc766062a04f7add43c2f7

## 规则

1. 比赛目标：参赛选手需要设计并训练四足机器人在指定赛道上行走，使其走得越远越好。
2. 比赛地图：比赛地图由组织者提供，包括赛道的长度、宽度和障碍物设置等信息。
3. 环境要求：参赛选手需要使用legged_gym作为基础环境，所有接口应保持与原有legged_gym一致。参赛选手可以选择使用强化学习或其他算法进行训练。
4. 比赛规则：
   1. 比赛使用的四足机器人需具备行走功能，并且在走道上保持平衡。
   2. 比赛终止条件为机器人的“base”触碰到地面或者到达终点最远端，视为比赛终止。
   3. 参赛选手需要提交训练好的模型及代码，或者使用自己电脑展示效果并录屏。
   4. 比赛结果将根据机器人在赛道上行走的距离来评判，行走距离越远者排名越靠前。
   5. 机器人可以是全自动的，也可以人为键盘控制机器人的行动。
   6. 不限制在环境中使用任何传感器
   7. 不限制学习算法的类型
   

## 强化学习基础

强化学习（Reinforcement Learning, RL）是一种让智能体（Agent）通过与环境交互来学习最佳行为策略的机器学习方法。智能体通过试错不断探索环境，每次采取行动后都会收到一个反馈（称为奖励），这个奖励告诉它行动的好坏。强化学习的目标是通过不断调整行为策略，最大化长期的累计奖励。

![让AI掌握星际争霸微操：中科院提出强化学习+课程迁移学习方法_凤凰科技](https://th.bing.com/th/id/R.8311e1e00fd70dd6a2a1ed749bff9066?rik=FghwHwWVTXBoxw&riu=http%3a%2f%2fp0.ifengimg.com%2fpmop%2f2018%2f0407%2fE99FE3D403A59F1F5E48922E0EAE972C7DE3B4B4_size25_w1080_h600.jpeg&ehk=FHxR0ukh7YbjE6f7gNQM4ccV9MgG44%2f4o%2fAVz%2b0BaHU%3d&risl=&pid=ImgRaw&r=0)

## 安装

我的环境是Ubuntu 20.04.6 LTS，pytroch版本是2.4.0+cu121，主要编程环境为VS Code。

### 安装准备

从库中拉开源项目文件，已经放到github上了，可以用clone，失败的话也可以直接"Download ZIP"

```sh
git clone https://github.com/jindadu00/legged_robot_competition
```

进入legged_robot_competition文件夹，打开终端，启动VS Code

### 安装legged_gym，isaacgym

这里我们要开始安装isaacgym，作为本项目的仿真环境。

原文档有些小错误，以下是原比赛文档中的内容

```sh
conda create -n legged_robot_parkour python=3.8
conda activate legged_robot_parkour
cd rsl_rl && git checkout v1.0.2 && pip install -e .
cd ..
cd isaacgym/python && pip install -e .
cd ..
cd legged_gym && pip install -e .
```

这里需要一点点linux知识(虽然不多)，或者求助于GPT，修改后依次运行下面的代码，最后回到根文件夹目录

```sh
conda create -n legged_robot_parkour python=3.8
conda activate legged_robot_parkour
cd rsl_rl && pip install -e .
cd .. 
cd isaacgym/python && pip install -e .
cd .. 
cd ..
cd legged_gym && pip install -e .
cd ..
```

## 快速入门，运行第一个demo

主程序在下面的文件夹下

```sh
cd legged_gym/legged_gym/scripts
```

运行train.py文件，

```python
conda activate legged_robot_parkour
cd legged_gym/legged_gym/scripts
python train.py --task=go2 --num_envs=64 --headless --max_iterations=50
```

我这里发现报错

```sh
AttributeError: module 'numpy' has no attribute 'float'.
```

这是因为numpy版本问题，如果遇到这个只需要搜索所有用到np.float的地方将其改为float

看到

```sh
###############################################################################
                       Learning iteration 0/6000                        

```

就说明开始正常训练了，可以通过设置--max_iterations来决定训练的轮数，这里方便起见选了--max_iterations=50。num_envs表示同时训练的数量，量力而行。我的GPU是4090D，选了4096，因为数字比较接近（bushi。

训练完成后，我们可以在legged_gym/logs/rough_go2找到这次训练保存的模型，这里的‘rough_go2’是默认的项目名称，要修改只需要添加--experiment_name修改为这次的名称，一般我会写本轮实验我要测试的奖励函数或者其他影响因素的名字。然后在-run_name写这个奖励值的具体数值。

对于已经训练好的模型，除了评估reward等量化指标，还需要用play.py在仿真环境中再测试一次。测试的代码如下

```sh
conda activate legged_robot_parkour
cd legged_gym/legged_gym/scripts

python play.py --task=go2 --num_envs=1  --checkpoint=50 --load_run=/path/to/your/project/legged_gym/logs/rough_go2/yourtime
```

现在，**你已经是个成熟的RLer了，能自己完成这个项目了**，以下是进阶内容。

## $\star$在巨人的肩膀上开始自研$\star$

要完成这个项目，学会自定义reward和observation是必不可少的。

### reward

本项目reward的具体值可以在legged_gym/legged_gym/envs/go2/go2_config.py中的102-119行开始修改，而这些函数的具体实现是在legged_gym/legged_gym/envs/go2/go2_robot.py文件中。

观察 \_prepare_reward_function() 函数的定义可以知道，这个函数会自动将所有命名为 “\_reward_name_” 

并且reward值不为0的都调用一遍，并在训练时候打印每一个epoch的奖励值。因此自定义奖励函数的步骤有两个，一是创建go2_config中的函数值，二是定义go2_robot中的奖励函数，并命名为 “\_reward_name_” 

要注意的点是reward函数需要返回(num_env, 1)这样维度的奖励，每写一个reward可以马上调试测试一下，即return 0但在reward函数中print这个奖励的shape维度。

### Observation

本项目的observation在legged_robot.py的self.obs_buf(在212行)，我们只需要模仿这个格式在cat函数里面加即可。主要注意的点同reward，要记录他的维度。此外要和策略网络的输入维度进行匹配，这里的设置在go2_config里面，这里的第7行和第9行的num_observations和num_privileged_obs。为了让狗能感知周围地形的高度，要将symmetric设置为False。

## 训练技巧

### 命令行修改参数

关于参数的信息可以在legged_gym/legged_gym/utils/helpers.py找到，如果要自定义命令行输入参数可以自己在custom_parameters加，不过并记得要在update_cfg_from_args中修改相关参数。

使用命令行改参数，可以方便我写脚本通宵测试今天写的奖励函数以及不同奖励值设置对结果的影响，脚本一般可以这么写，要改成自己的路径。这个脚本的含义就是系统会依次执行这里的每一行，作为一个例子，我写了个奖励函数命名为limbo，然后分别测试它等于-0.5, -1.5等的情况下控制算法的表现。

```sh
#! /bin/bash
cd /path/to/your/project/legged_gym/legged_gym/scripts
python train.py --task=go2 --num_envs=4096 --headless --resume  --model_dir=/path/to/your/project/legged_gym/logs/move_back/Aug31_22-49-13_0_015 --experiment_name=limbo  --max_iterations=2000 --goal_position_x=75  --run_name=0_5 --move_back=-0.015 --goal_pos=200.0 --limbo=-0.5
python train.py --task=go2 --num_envs=4096 --headless --resume  --model_dir=/path/to/your/project/legged_gym/logs/move_back/Aug31_22-49-13_0_015 --experiment_name=limbo  --max_iterations=2000 --goal_position_x=75  --run_name=1_5 --move_back=-0.015 --goal_pos=200.0 --limbo=-1.5
```

### 记录实验结果

我这里还在train函数中将本次运行的所有reward值打印并写入一个txt文件保存，并且在on_policy_runner.py文件中修改log函数，使其能将最新的实验结果记录下来，结合这两者将方便后续分析实验结果。

### 可视化训练

训练过程可以用tensorboard来可视化所有训练参数

```sh
tensorboard --logdir=/path/to/your/project/legged_gym/logs/xxx
```

### 奖励函数值的调整方向

奖励函数可以大致分为三类，**任务奖励**、**增强辅助奖励**和**固定辅助奖励**

任务奖励是直接与强化学习任务目标相关的奖励，激励机器人完成特定的任务。在这里就是追踪速率，这是机器狗要完成的首要目标，应设计为一个合理的正值。

增强辅助奖励是机器狗达成最终目标需要学会的阶段性子任务，根据经验给予阶段性奖励引导机器狗在前期更快地朝正确方向学习减少探索的错误行为。比如机器狗要学会保持平衡，就要对摔倒进行惩罚。

固定辅助奖励是一种固定的、与任务无关的奖励，用来限制机器人执行不期望的行为或引导其遵守一定的行为规范。比如对机器狗的能耗进行惩罚。

在训练过程中，一个担忧是**机器人可能会放弃任务或在辅助目标的惩罚压倒任务奖励时选择提前终止。**为了解决这个问题，我们要让奖励函数在大多数时候都保持为一个正值，如果始终都是负的，则适当地减小惩罚的值。

### 初始状态的设置

通过设置机器狗的出生位置，我们可以增加随机性，让机器狗在前期就见到足够多的地形，也可以专门训练一种地形策略。这在go2_robot.py的745-752行进行修改。

### 状态的获取

- self.dof_pos维度是(env_num, 12)，12为关节的个数，需要找到具体的indices来设置，例如hip_indices可以通过下面的代码创建

```
hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
for i, name in enumerate(hip_names):
self.hip_indices[i] = self.dof_names.index(name)
```

- self.root_states表示基座的状态，维度为(env_num, 12)，其中self.root_states[:, :3]表示基座位置xyz，self.root_states[:, 3:7]表示基座角位置用四元数表示,self.root_states[:, 7:10]表示基座线速度，self.root_states[:, 10:13]表示基座角速度。
- self.rigid_body_states表示组成机器狗的所有模块部位的状态，它有三个维度，第一个维度同样是env_num，第二个维度是我们需要如上面的hip_indices一样找到要找的模块对应的编号，例如self.feet_indices记录了所有脚的编号，第三个维度和self.root_states的第二个维度相同。
- self.contact_forces前两个维度和self.rigid_body_states相同，第三个维度表示xyz方向上的力。

## 策略技巧

### 迁移学习

从简单到难，在完成一个阶段性任务后在原来的模型基础上继续训练。这里我自己写了个读取模型继续训练的参数。

即修改了make_alg_runner函数读取模型路径的方式，直接导入模型的路径，不用log_root和load_run等参数进行拼接，代码如下

```python
if resume:
    if resume_path is None:
        # load previously trained model
        if train_cfg.runner.model_dir:
            resume_path = get_load_path(train_cfg.runner.model_dir, load_run="", checkpoint=train_cfg.runner.checkpoint)
        else:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)
```

然后继续训练的代码示例如下

```sh
conda activate legged_robot_parkour
cd legged_gym/legged_gym/scripts
python train.py --task=go2 --num_envs=4096 --headless --resume   --experiment_name=whole  --max_iterations=100000  --run_name=from_Sep18_15-56-08_ --model_dir=/path/to/your/project/legged_gym/logs/xxxx --checkpoint=300
```

设计完成全部就加大量分数，让机器狗一步步实现跑完全程的目标。我采取的训练策略分为如下几步

- 能到梅花桩前

- 简化版梅花桩跳一下就结束

- 简化版梅花桩跳两下就结束

- 简化版梅花桩跳全部并出来

- 梅花桩跳全部并出来

- 走完全程

我这里还对梅花桩地形作了修改，原地形梅花桩不仅小还有高差，多次实验发现狗学会了一些奇技淫巧来跳跃拿分但是就是走不完全程，于是就让他从简单的开始，加大梅花桩面积让他能在梅花桩上稳稳地站住，同时让所有梅花桩都一样高。

### 分层策略

要让机器狗学会跨越多种地形，尤其是最后一关梅花桩其实难度还是比较大的。我在探索过程中发现从头训练机器狗到梅花桩前，以及单独训练机器狗跨过梅花桩这两个任务都完成得令人满意，但是一旦用同一个网络训练全程就遇到困难，适用于梅花桩地形的策略在前面理论上更简单的地形做的不好，相反狗只见过前面地形则跳不过梅花桩，我认为这是本项目最难的地方。

而我对此给出的解决方案有取巧的成分，采用分层策略的思想，训练两套参数，然后在适当的情况下进行策略的切换。在这里就是判断地形的种类，然后调用适用于这个地形的策略。好比是训练大模型有难度，但专业化的小模型能接受。这样专业化训练的思路大大减小了模型训练的难度，同时也能完成任务。

## My reward and observation

这个开源项目大多数奖励函数都写得很好，由于这个项目任务的特殊性，我自己在训练过程中自己写了很多函数，虽然大多数被证明用处不大反而会影响训练，但也有几点我修改的被证实有用的主要reward和observation可以分享。

奖励和观测不用太多，质量大于数量，大道至简，less is more。

### Reward

#### tracking_goal_vel

抛弃对固定速度的追踪奖励作为任务奖励，也就是tracking_lin_vel和tracking_ang_vel。因为我在用它作为任务奖励训练时候机器狗会陷入到波浪地形中出不来，是个局部最优解。取而代之我选用了目标点追踪的奖励函数，即人为设立一系列目标点，到达一个目标点就换下一个目标点。奖励函数设计如下，首先定义$d$为目前狗基座到下一个目标点的向量的水平单位分量
$$
d=\frac{p-x}{\left|\left| p-x\right|\right| }
$$
这里的奖励函数设计为当前狗基座的水平速度在$d$上的投影，且不超过$v_{cmd}$ ，后者可以控制狗的速度尽量不超过控制的目标速度。
$$
r=min(<v,d>,v_{cmd})
$$
这个设计的好处是狗能从中获得的最大的奖励被设置了一个上限，无法通过来回走反复刷分，解决了机器狗会陷入到波浪地形中出不来的问题

#### reach_all_goal

要一步步训练机器狗完成越来越难的目标，我将对goal的位置和数量进行人为控制。当狗完成我当前给他的全部目标时候就提前终止，然后给予一个很大的奖励，让它记住这被狠狠奖励的感觉。

#### feet_edge

在梅花桩区域，狗踩在梅花桩区域是不稳定的，根据我训练过程中的观察有时候狗还会通过蹬梅花桩的边缘获得一个不错的跳跃速度，但这么做只能跳一下。因此脚踩在梅花桩边缘的行为应该被惩罚，我的代码如下，其中edge_mask是对地形高度边缘检测得到的边缘掩码。

```python
def _reward_feet_edge(self):
    foot_pos_xy=self.rigid_body_states[:, self.feet_indices, :2]

    indices = (foot_pos_xy / 0.25).long()  # (num_envs, 4, 2)

    indices[:, :, 0] = indices[:, :, 0].clamp(0, self.height_samples.shape[0] - 1) 
    indices[:, :, 1] = indices[:, :, 1].clamp(0, self.height_samples.shape[1] - 1) 

    feet_at_edge = self.edge_mask[indices[:, :, 0], indices[:, :, 1]]

    self.feet_at_edge = self.contact_filt & feet_at_edge
    rew = torch.sum(self.feet_at_edge, dim=-1)
    return rew
```

### Observation

#### measured_heights

为了让狗更好地通过梅花桩区域，将measured_points_x向前延伸，使其能够观察到对岸的台阶而不是全部都是深渊，可以适当减少对背后的观察。



#### delta_yaw

告诉机器狗当前需要将身体调整多少偏航角才能朝向目标点
$$
reward_{yaw} = yaw_{target} - yaw
$$

#### env_class

因为梅花桩这个地形很复杂，希望告诉机器狗现在是什么地形。我采用独热编码，如果是梅花桩就输入(1, 0)，如果不是就输入(0, 1)。

#### contact

由于机器狗要跳跃，告诉机器狗四条腿和地面的接触情况，可以让它在空中更好做动作。

#### 历史观察回放

将除去对地形的观测的全部历史的观察信息放入一个历史状态序列，使得机器狗获得了记忆能力，能更好完成跨越梅花桩等高难动作。
