import random
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for adaptive traffic signal control.

    State:
        (queue_state, waiting_state, phase)

    Action:
        0 -> KEEP current phase
        1 -> CHANGE to next phase
    """

    def __init__(
        self,
        learning_rate=0.10,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.97,
        actions=None
    ):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.actions = (
            list(actions)
            if actions is not None
            else [0, 1]
        )

        self.q_table = defaultdict(
            self._default_q_values
        )

    # ========================================================
    # DEFAULT Q VALUES
    # ========================================================

    def _default_q_values(self):

        return {
            action: 0.0
            for action in self.actions
        }

    # ========================================================
    # STATE REPRESENTATION
    # ========================================================

    def get_state(
        self,
        queue_length,
        waiting_time,
        phase
    ):
        """
        Discrete RL state.

        Queue:
            0 = 0-2
            1 = 3-5
            2 = 6-10
            3 = 11-20
            4 = >20

        Waiting:
            0 = 0-5
            1 = 6-15
            2 = 16-30
            3 = 31-60
            4 = >60

        Phase:
            0, 1, 2, 3
        """

        try:
            queue_length = max(
                0.0,
                float(queue_length)
            )
        except (TypeError, ValueError):
            queue_length = 0.0

        try:
            waiting_time = max(
                0.0,
                float(waiting_time)
            )
        except (TypeError, ValueError):
            waiting_time = 0.0

        try:
            phase = int(phase)
        except (TypeError, ValueError):
            phase = 0

        # Queue state
        if queue_length <= 2:
            queue_state = 0
        elif queue_length <= 5:
            queue_state = 1
        elif queue_length <= 10:
            queue_state = 2
        elif queue_length <= 20:
            queue_state = 3
        else:
            queue_state = 4

        # Waiting state
        if waiting_time <= 5:
            waiting_state = 0
        elif waiting_time <= 15:
            waiting_state = 1
        elif waiting_time <= 30:
            waiting_state = 2
        elif waiting_time <= 60:
            waiting_state = 3
        else:
            waiting_state = 4

        phase_state = phase % 4

        return (
            queue_state,
            waiting_state,
            phase_state
        )

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(self, state):

        if (
            self.epsilon > 0
            and random.random() < self.epsilon
        ):
            return random.choice(self.actions)

        return self.get_best_action(state)

    # ========================================================
    # GREEDY ACTION
    # ========================================================

    def get_best_action(self, state):
        """
        Deterministic greedy action.

        If Q-values are tied, KEEP is preferred.
        This makes evaluation reproducible.
        """

        q_values = self.q_table[state]

        max_q = max(
            q_values.values()
        )

        best_actions = [
            action
            for action, value in q_values.items()
            if value == max_q
        ]

        # Prefer KEEP on a tie
        if 0 in best_actions:
            return 0

        return best_actions[0]

    # ========================================================
    # Q-LEARNING UPDATE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        current_q = (
            self.q_table[state][action]
        )

        next_q = max(
            self.q_table[next_state].values()
        )

        target = (
            reward
            + self.discount_factor
            * next_q
        )

        self.q_table[state][action] = (
            current_q
            + self.learning_rate
            * (
                target
                - current_q
            )
        )

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay
        )

    # ========================================================
    # EVALUATION MODE
    # ========================================================

    def set_evaluation_mode(self):

        self.epsilon = 0.0

    # ========================================================
    # TRAINING MODE
    # ========================================================

    def set_training_mode(
        self,
        epsilon=None
    ):

        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )

    # ========================================================
    # Q-TABLE INFORMATION
    # ========================================================

    def get_q_table_size(self):

        return len(self.q_table)

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, filename):

        q_table_to_save = {}

        for state, actions in self.q_table.items():

            if isinstance(actions, dict):

                q_table_to_save[state] = dict(
                    actions
                )

            elif isinstance(
                actions,
                (list, tuple)
            ):

                q_table_to_save[state] = list(
                    actions
                )

            else:

                q_table_to_save[state] = actions

        with open(
            filename,
            "wb"
        ) as file:

            pickle.dump(
                q_table_to_save,
                file
            )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, filename):

        with open(
            filename,
            "rb"
        ) as file:

            loaded_table = pickle.load(
                file
            )

        self.q_table = defaultdict(
            self._default_q_values,
            loaded_table
        )

        # Make sure every state has both actions
        for state in list(
            self.q_table.keys()
        ):

            for action in self.actions:

                if action not in self.q_table[state]:

                    self.q_table[state][action] = 0.0

        return len(
            self.q_table
        )