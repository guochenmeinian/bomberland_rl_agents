[![Engine Version](https://img.shields.io/badge/engine%20ver.-2381-blue)](#release-notes)

# Bomberland Project

[![GitHub Repo](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/guochenmeinian/bomberland_rl_agents.git) | [![Website](https://img.shields.io/badge/Website-Online-blue?logo=google-chrome)](https://guochenmeinian.github.io/bomberland_rl_report/)

## About

[Bomberland](https://www.gocoder.one/bomberland) is a multi-agent AI competition inspired by the classic console game Bomberman.

Teams build intelligent agents using strategies from tree search to deep reinforcement learning. The goal is to compete in a 2D grid world collecting power-ups and placing explosives to take your opponent down.

This repo contains our implementation for the project.

![Bomberland multi-agent environment](./engine/bomberland-ui/src/source-filesystem/docs/2-environment-overview/bomberland-preview.gif "Bomberland")

## Usage

### Basic usage

See: [Documentation](https://www.gocoder.one/docs)

1. Clone or download this repo (including both `base-compose.yml` and `docker-compose.yml` files).

2. Build the Docker Image:
    ```
    cd engine
    docker build -t bomberland:011a277 -f bomberland-engine/Dockerfile . 
    ```

3. To connect agents and run a game instance, run from the root directory:
    ```
    docker-compose up --abort-on-container-exit --force-recreate
    ```

4. While the engine is running, access the client by going to `http://localhost:3000/` in your browser (may be different depending on your settings).

5. From the client, you can connect as a `spectator` or `agent` (to play as a human player)

### Open AI gym wrapper (experimental)

`docker-compose -f open-ai-gym-wrapper-compose.yml up --force-recreate --abort-on-container-exit`


## Resources

### Starter kits

| Kit                 | Link                                                                           | Description                                        | Up-to-date? | Contributed by                          |
| ------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- | ----------- | --------------------------------------- |
| Python3             | [Link](https://github.com/CoderOneHQ/bomberland/tree/master/agents/python3)    | Basic Python3 starter                              | ✅          | Coder One                               |
| Python3-fwd         | [Link](https://github.com/CoderOneHQ/bomberland/tree/master/agents/python3)    | Includes example for using forward model simulator | ✅          | Coder One                               |
| Python3-gym-wrapper | [Link](https://github.com/CoderOneHQ/bomberland/tree/master/agents/python3)    | Open AI Gym wrapper                                | ✅          | Coder One                               |

Official Discord Community: [Link](https://discord.gg/Hd8TRFKsDa).

Please let us know of any bugs or suggestions by [raising an Issue](https://github.com/guochenmeinian/bomberland_rl_agents/issues).
