import os
import csv
import time

from sumo_runner import SUMORunner
from rl_agent import QLearningAgent

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON,
    EPSILON_MIN,
    EPSILON_DECAY,
    MIN_GREEN_TIME,
    TOTAL_STEPS,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPISODES = 100
STEPS_PER_EPISODE = TOTAL_STEPS

RESULTS_DIR = "results"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "rl_results.csv"
)

Q_TABLE_FILE = os.path.join(
    RESULTS_DIR,
    "q_table.pkl"
)

TRAFFIC_DATA_FILE = os.path.join(
    RESULTS_DIR,
    "traffic_data.csv"
)


# ============================================================
# TRAFFIC SIGNAL CONTROL
# ============================================================

PHASES = 4

MIN_PHASE_TIME = MIN_GREEN_TIME

ACTION_COOLDOWN = 5


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CLEAR SCREEN
# ============================================================

def clear_screen():

    os.system(
        "cls" if os.name == "nt" else "clear"
    )


# ============================================================
# GET RL STATE
# ============================================================

def get_state(
    runner,
    agent
):

    traffic = runner.get_traffic_state()

    return agent.get_state(
        traffic["queue"],
        traffic["waiting_time"],
        traffic["phase"]
    )


# ============================================================
# CALCULATE REWARD
# ============================================================

def calculate_reward(
    previous_traffic,
    current_traffic
):

    previous_queue = (
        previous_traffic["queue"]
    )

    current_queue = (
        current_traffic["queue"]
    )

    previous_waiting = (
        previous_traffic["waiting_time"]
    )

    current_waiting = (
        current_traffic["waiting_time"]
    )

    previous_throughput = (
        previous_traffic["step_throughput"]
    )

    current_throughput = (
        current_traffic["step_throughput"]
    )

    # --------------------------------------------------------
    # TRAFFIC CHANGES
    # --------------------------------------------------------

    queue_change = (
        previous_queue
        - current_queue
    )

    waiting_change = (
        previous_waiting
        - current_waiting
    )

    throughput_change = (
        current_throughput
        - previous_throughput
    )

    # --------------------------------------------------------
    # REWARD
    # --------------------------------------------------------

    reward = 0.0

    reward += (
        queue_change * 1.5
    )

    reward += (
        waiting_change * 0.25
    )

    reward += (
        throughput_change * 5.0
    )

    reward -= (
        current_queue * 0.5
    )

    reward -= (
        current_waiting * 0.05
    )

    # --------------------------------------------------------
    # REWARD BOUND
    # --------------------------------------------------------

    reward = max(
        -10.0,
        min(
            10.0,
            reward
        )
    )

    return reward


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    with open(
        RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Episode",
            "Total_Reward",
            "Steps",
            "Epsilon",
            "Q_Table_States",
            "Signal_Changes"
        ])

        for result in results:

            writer.writerow([
                result["episode"],
                result["reward"],
                result["steps"],
                result["epsilon"],
                result["q_states"],
                result["signal_changes"]
            ])


# ============================================================
# SAVE TRAFFIC DATA
# ============================================================

def save_traffic_data(
    traffic_records
):

    if not traffic_records:

        print(
            "No traffic data available to save."
        )

        return None

    with open(
        TRAFFIC_DATA_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode",
                "step",
                "vehicles",
                "queue",
                "waiting_time",
                "average_speed",
                "phase",
                "throughput"
            ]
        )

        writer.writeheader()

        writer.writerows(
            traffic_records
        )

    return TRAFFIC_DATA_FILE


# ============================================================
# SAVE Q-TABLE
# ============================================================

def save_q_table(agent):

    agent.save(
        Q_TABLE_FILE
    )

    return agent.get_q_table_size()


# ============================================================
# DECAY EPSILON
# ============================================================

def decay_epsilon(agent):

    agent.decay_epsilon()


# ============================================================
# MAIN
# ============================================================

def main():

    clear_screen()

    print("=" * 60)
    print("RL TRAINING")
    print("=" * 60)

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        f"Total planned steps: "
        f"{EPISODES * STEPS_PER_EPISODE}"
    )

    print()

    print("RL parameters:")

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Discount factor: {DISCOUNT_FACTOR}"
    )

    print(
        f"Initial epsilon: {EPSILON}"
    )

    print(
        f"Minimum epsilon: {EPSILON_MIN}"
    )

    print(
        f"Epsilon decay: {EPSILON_DECAY}"
    )

    print()

    print("Action space:")

    print(
        "0 = KEEP current phase"
    )

    print(
        "1 = CHANGE to next phase"
    )

    print()

    print(
        f"SUMO phases: {PHASES}"
    )

    print(
        f"Minimum phase time: "
        f"{MIN_PHASE_TIME}s"
    )

    print(
        f"Action cooldown: "
        f"{ACTION_COOLDOWN}s"
    )

    print()


    # ========================================================
    # CREATE AGENT
    # ========================================================

    agent = QLearningAgent(
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )


    results = []
    traffic_records = []


    # ========================================================
    # TRAINING
    # ========================================================

    for episode in range(
        1,
        EPISODES + 1
    ):

        clear_screen()

        print("=" * 60)

        print(
            f"RL TRAINING - EPISODE "
            f"{episode}/{EPISODES}"
        )

        print("=" * 60)

        print(
            f"Current epsilon: "
            f"{agent.epsilon:.4f}"
        )

        print()


        runner = SUMORunner(
            use_gui=False,
            step_length=1
        )

        total_reward = 0.0
        steps_completed = 0
        signal_changes = 0

        phase_timer = 0
        cooldown_timer = 0


        try:

            # ------------------------------------------------
            # START SUMO
            # ------------------------------------------------

            runner.start()


            # ------------------------------------------------
            # INITIAL TRAFFIC
            # ------------------------------------------------

            initial = (
                runner.get_traffic_state()
            )

            previous_traffic = {

                "vehicles":
                    initial["vehicles"],

                "queue":
                    initial["queue"],

                "waiting_time":
                    initial["waiting_time"],

                "throughput":
                    initial["throughput"],

                "step_throughput":
                    0,

                "phase":
                    initial["phase"]
            }


            # ------------------------------------------------
            # INITIAL STATE
            # ------------------------------------------------

            state = agent.get_state(
                initial["queue"],
                initial["waiting_time"],
                initial["phase"]
            )


            # =================================================
            # SIMULATION LOOP
            # =================================================

            for step in range(
                STEPS_PER_EPISODE
            ):

                phase_timer += 1

                if cooldown_timer > 0:
                    cooldown_timer -= 1


                # ------------------------------------------------
                # SELECT ACTION
                # ------------------------------------------------

                action = agent.choose_action(
                    state
                )

                if action not in (0, 1):
                    action = 0


                # ------------------------------------------------
                # CHECK CHANGE
                # ------------------------------------------------

                can_change = (
                    phase_timer >= MIN_PHASE_TIME
                    and cooldown_timer == 0
                )


                # ------------------------------------------------
                # APPLY ACTION
                # ------------------------------------------------

                if (
                    action == 1
                    and can_change
                ):

                    current_phase = (
                        runner.get_phase()
                    )

                    next_phase = (
                        current_phase + 1
                    ) % PHASES

                    runner.set_phase(
                        next_phase
                    )

                    phase_timer = 0

                    cooldown_timer = (
                        ACTION_COOLDOWN
                    )

                    signal_changes += 1


                # ------------------------------------------------
                # ADVANCE SUMO
                # ------------------------------------------------

                if not runner.step():
                    break

                steps_completed += 1


                # ------------------------------------------------
                # GET CURRENT TRAFFIC
                # ------------------------------------------------

                current = (
                    runner.get_traffic_state()
                )

                step_throughput = (
                    runner.get_step_throughput()
                )

                current_traffic = {

                    "vehicles":
                        current["vehicles"],

                    "queue":
                        current["queue"],

                    "waiting_time":
                        current["waiting_time"],

                    "average_speed":
                        current["average_speed"],

                    "phase":
                        current["phase"],

                    "throughput":
                        current["throughput"],

                    "step_throughput":
                        step_throughput
                }


                # ------------------------------------------------
                # NEXT STATE
                # ------------------------------------------------

                next_state = agent.get_state(
                    current_traffic["queue"],
                    current_traffic["waiting_time"],
                    current_traffic["phase"]
                )


                # ------------------------------------------------
                # REWARD
                # ------------------------------------------------

                reward = calculate_reward(
                    previous_traffic,
                    current_traffic
                )

                total_reward += reward


                # ------------------------------------------------
                # Q UPDATE
                # ------------------------------------------------

                agent.update(
                    state,
                    action,
                    reward,
                    next_state
                )


                # ------------------------------------------------
                # SAVE DATA
                # ------------------------------------------------

                traffic_records.append({

                    "episode":
                        episode,

                    "step":
                        steps_completed,

                    "vehicles":
                        current_traffic["vehicles"],

                    "queue":
                        current_traffic["queue"],

                    "waiting_time":
                        current_traffic["waiting_time"],

                    "average_speed":
                        current_traffic["average_speed"],

                    "phase":
                        current_traffic["phase"],

                    "throughput":
                        current_traffic["throughput"]
                })


                state = next_state

                previous_traffic = (
                    current_traffic
                )


                time.sleep(
                    0.001
                )


        except Exception as e:

            print()

            print(
                f"Episode {episode} error:"
            )

            print(e)


        finally:

            runner.close()


        # ====================================================
        # EPSILON DECAY
        # ====================================================

        decay_epsilon(
            agent
        )


        epsilon = (
            agent.epsilon
        )

        q_states = (
            agent.get_q_table_size()
        )


        results.append({

            "episode":
                episode,

            "reward":
                total_reward,

            "steps":
                steps_completed,

            "epsilon":
                epsilon,

            "q_states":
                q_states,

            "signal_changes":
                signal_changes
        })


        print()

        print("-" * 60)

        print(
            f"Episode {episode} completed"
        )

        print(
            f"Reward: {total_reward:.2f}"
        )

        print(
            f"Steps: {steps_completed}"
        )

        print(
            f"Q-table states: {q_states}"
        )

        print(
            f"Signal changes: {signal_changes}"
        )

        print(
            f"Next epsilon: {epsilon:.4f}"
        )

        print("-" * 60)


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()

    print(
        "Saving training results..."
    )

    save_results(
        results
    )

    traffic_file = (
        save_traffic_data(
            traffic_records
        )
    )

    try:

        saved_states = (
            save_q_table(
                agent
            )
        )

        print()

        print(
            "Q-table successfully saved."
        )

        print(
            f"Q-table states: "
            f"{saved_states}"
        )

    except Exception as e:

        print()

        print(
            "WARNING: Q-table could not be saved."
        )

        print(e)


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        f"Total simulation steps: "
        f"{len(traffic_records)}"
    )

    print(
        f"Q-table states: "
        f"{len(agent.q_table)}"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.4f}"
    )

    print()

    print(
        "Traffic dataset saved to:"
    )

    print(
        traffic_file
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print()

    print(
        "Q-table saved to:"
    )

    print(
        Q_TABLE_FILE
    )

    print()


if __name__ == "__main__":

    main()