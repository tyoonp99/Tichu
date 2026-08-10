from gymnasium.envs.registration import register

register(
    id='tichu_multiplayer-v0',
    entry_point='gym_tichu.envs:TichuMultiplayerEnv',
    reward_threshold=200.0,
    nondeterministic=True,
    disable_env_checker=True,
    order_enforce=False,
)

register(
    id='tichu_singleplayer-v0',
    entry_point='gym_tichu.envs:TichuSinglePlayerEnv',
    reward_threshold=200.0,
    nondeterministic=True,
    disable_env_checker=True,
    order_enforce=False,
)
