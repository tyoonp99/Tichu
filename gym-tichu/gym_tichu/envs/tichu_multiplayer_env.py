
import gymnasium as gym
import logging
import random
from typing import Any, Optional, Tuple
from profilehooks import timecall

from .internals import *
from .internals.error import IllegalActionError, LogicError


logger = logging.getLogger(__name__)


class TichuMultiplayerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, illegal_move_mode: str='raise', verbose: bool=True, render_mode: Optional[str]=None):
        """
        :param illegal_move_mode: 'raise' or 'loose'. If 'raise' an exception is raised, 'loose' and the team looses 200:0
        :param verbose: if True, logs to the info log, if false, logs to the debug log
        """
        assert illegal_move_mode in ['raise'], "'loose' is not yet implemented"  # ['raise', 'loose']
        assert render_mode is None, "Rendering is not implemented"

        super().__init__()

        self._current_state = None
        self.verbose = verbose
        self.render_mode = render_mode

    def step(self, action: Any)->Tuple[TichuState, Tuple[int, int, int, int], bool, bool, dict]:
        # logger.debug("step with action {}".format(action))

        state = self._current_state.next_state(action)
        self._current_state = state

        terminated = state.is_terminal()
        points = state.count_points() if terminated else (0, 0, 0, 0)
        if terminated:
            state = state.change(history=state.history.add_last_state(state))
            self._current_state = state
        return state, points, terminated, False, dict()

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None)->Tuple[InitialState, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self._current_state = InitialState()
        return self._current_state, dict()

    def render(self):
        # print("RENDER: ", self._current_state)
        pass

    def _log(self, message, *args, **kwargs):
        if self.verbose:
            logger.info(message, *args, **kwargs)
        else:
            logger.debug(message, *args, **kwargs)


class TichuSinglePlayerEnv(TichuMultiplayerEnv):
    """
    Environment for one player. The other players can be set with the 'configure' method
    """

    def __init__(self, verbose: bool=True, render_mode: Optional[str]=None):
        """
        :param verbose: if True, logs to the info log, if false, logs to the debug log
        """

        self._agents = (None, None, None, None)  # set with the 'configure' method
        super().__init__(verbose=verbose, render_mode=render_mode)

    @timecall(immediate=False)
    def step(self, action: PlayerAction)->Tuple[BaseTichuState, int, bool, bool, dict]:
        assert self._agents[2] is not None

        try:
            state, reward, terminated, truncated, info = super().step(action)
            # logger.debug("Legal Action! {}".format(action))
        except IllegalActionError:
            logger.debug("Illegal Action! {}, legal are: {}".format(action, self._current_state.possible_actions_list))
            return self._current_state, -500, True, False, {'illegalAction': action}

        player_state, reward, terminated, info = self._forward_to_player(state)

        assert terminated or player_state.player_pos == 0
        # if done:
        #     logger.debug("TichuSinglePlayerAgainstRandomEnv, Final State: {}".format(state))

        assert terminated == player_state.is_terminal()
        assert terminated or player_state.player_pos == 0, str(player_state)
        return player_state, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None)->Tuple[BaseTichuState, dict]:
        # init
        _ = super().reset(seed=seed, options=options)
        # NO grand tichu
        _ = super().step({})
        # No (normal) tichu now
        _ = super().step({})
        # No trading cards
        state, _, _, _, _ = super().step([])
        # forward to player
        player_state, _, _, info = self._forward_to_player(state)
        return player_state, info

    def _forward_to_player(self, state: BaseTichuState)->Tuple[BaseTichuState, int, bool, dict]:
        """
        :return: The next state in which the player 0 can play a Combiantion.
        """
        # logger.debug("Forwarding to player 0")

        if state.is_terminal():
            # logger.debug("State is already terminal, Nothing to forward.")
            return state, state.count_points()[0], True, dict()

        curr_state = state
        curr_reward = (0, 0, 0, 0)
        done = False
        info = dict()

        first_action = state.possible_actions_list[0]
        # Note: for both tichu and wish action, state.player_pos is not the same as action.player_pos, it is the pos of the next player to play a combination

        while not isinstance(first_action, (PassAction, PassBombAction, PlayCombination)) or first_action.player_pos != 0:
            # logger.debug("state: {}".format(state))
            # No TICHU
            if isinstance(first_action, TichuAction):
                no_tichu_action = next(filter(lambda act: act.announce is False, curr_state.possible_actions_list))
                curr_state, curr_reward, done, _, info = super().step(no_tichu_action)

            # No WISH
            elif isinstance(first_action, WishAction):
                no_wish_action = WishAction(player_pos=first_action.player_pos, wish=None)
                curr_state, curr_reward, done, _, info = super().step(no_wish_action)

            # TRICK ENDS
            elif isinstance(first_action, WinTrickAction):
                curr_state, curr_reward, done, _, info = super().step(first_action)

            # Play Combination
            elif isinstance(first_action, (PassAction, PassBombAction, PlayCombination)):
                assert curr_state.player_pos != 0
                # other agents choose action

                action = self._agents[curr_state.player_pos].action(state=curr_state)
                curr_state, curr_reward, done, _, info = super().step(action)

            else:
                raise LogicError()

            if done:
                # logger.debug("State is terminal -> break out of forward to player")
                # logger.debug("Final State: {}".format(curr_state))
                break

            first_action = curr_state.possible_actions_list[0]

        assert done or curr_state.player_pos == 0
        assert done == curr_state.is_terminal()
        return curr_state, curr_reward[0], done, info

    def configure(self, *args, other_agents: Tuple[Any, Any, Any], **kwargs):
        """
        :param other_agents: the 3 agents to play against. Note that other_agents[1] is the teammate.
        :return: 
        """
        assert len(other_agents) == 3
        self._agents = (None,) + tuple(other_agents)
