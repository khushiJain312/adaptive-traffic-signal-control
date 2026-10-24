
import os
import time

from config import TOTAL_STEPS
from sumo_runner import SUMORunner
from rl_agent import QLearningAgent


def main():

    print("=" * 60)
    print("RL TRAFFIC SIGNAL DEMONSTRATION")
    print("=" * 60)

    # ========================================================
    # LOAD TRAINED Q-TABLE
    # ========================================================

    q_table_file = os.path.join(
        "results",
        "q_table.pkl"
    )

    print("Loading trained Q-table...")

    agent = QLearningAgent()

    if not os.path.exists(q_table_file):

        print("ERROR: Q-table not found:")
        print(os.path.abspath(q_table_file))

        return

    agent.load(q_table_file)

    print("Q-table loaded successfully.")
    print(
        "Q-table states:",
        agent.get_q_table_size()
    )

    # ========================================================
    # START SUMO-GUI
    # ========================================================

    runner = SUMORunner(
        use_gui=True,
        step_length=1
    )

    try:

        runner.start()

        print()
        print("=" * 60)
        print("RL DEMONSTRATION STARTED")
        print("=" * 60)
        print("SUMO-GUI is running.")
        print("The trained Q-learning agent is controlling")
        print("the traffic signal.")
        print()
        print("Press CTRL+C to stop.")
        print("=" * 60)

        total_reward = 0.0

        # ====================================================
        # DEMONSTRATION LOOP
        # ====================================================

        for step in range(TOTAL_STEPS):

            if not runner.is_running():

                print()
                print("Simulation finished.")

                break

            # ------------------------------------------------
            # GET CURRENT TRAFFIC STATE
            # ------------------------------------------------

            traffic_state = runner.get_state()

            state = agent.get_state(
                traffic_state["queue_length"],
                traffic_state["waiting_time"],
                traffic_state["phase"]
            )

            # ------------------------------------------------
            # CHOOSE BEST LEARNED ACTION
            # ------------------------------------------------

            action = agent.get_best_action(
                state
            )

            # ------------------------------------------------
            # APPLY RL ACTION
            # ------------------------------------------------

            runner.apply_action(
                action
            )

            # ------------------------------------------------
            # ADVANCE SUMO
            # ------------------------------------------------

            runner.step()

            # ------------------------------------------------
            # GET NEW STATE
            # ------------------------------------------------

            new_state = runner.get_state()

            # ------------------------------------------------
            # CALCULATE REWARD
            # ------------------------------------------------

            throughput = runner.get_throughput()

            reward = agent.calculate_reward(
                new_state["queue_length"],
                new_state["waiting_time"],
                throughput
            )

            total_reward += reward

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            action_name = (
                "KEEP PHASE"
                if action == 0
                else "CHANGE PHASE"
            )

            print(
                f"Step: {step + 1:4d} | "
                f"Vehicles: "
                f"{runner.get_vehicle_count():3d} | "
                f"Queue: "
                f"{new_state['queue_length']:3d} | "
                f"Waiting: "
                f"{new_state['waiting_time']:7.2f} | "
                f"Phase: "
                f"{new_state['phase']:2d} | "
                f"Action: "
                f"{action_name:12s} | "
                f"Reward: "
                f"{reward:8.2f}"
            )

            time.sleep(0.05)

        # ====================================================
        # FINAL RESULTS
        # ====================================================

        print()
        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)

        print(
            f"Total reward: {total_reward:.2f}"
        )

        print(
            f"Q-table states: "
            f"{agent.get_q_table_size()}"
        )

        print(
            f"Final epsilon: "
            f"{agent.epsilon:.4f}"
        )

        print("=" * 60)

    except KeyboardInterrupt:

        print()
        print("Demonstration stopped by user.")

    except Exception as error:

        print()
        print("ERROR:")
        print(error)

    finally:

        runner.close()

        print()
        print("=" * 60)
        print("SUMO CLOSED")
        print("=" * 60)


if __name__ == "__main__":

    main()
