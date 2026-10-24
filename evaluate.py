import os
import csv
import pickle
import time

from sumo_runner import SUMORunner


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

EPISODES = 30
STEPS_PER_EPISODE = 500

RESULTS_DIR = "results"

Q_TABLE_FILE = os.path.join(
    RESULTS_DIR,
    "q_table.pkl"
)

EVALUATION_FILE = os.path.join(
    RESULTS_DIR,
    "rl_evaluation.csv"
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

def get_state(runner):

    traffic = runner.get_traffic_state()

    queue = traffic["queue"]
    waiting_time = traffic["waiting_time"]
    phase = traffic["phase"]

    # --------------------------------------------------------
    # Queue state
    # --------------------------------------------------------

    if queue <= 2:

        queue_state = 0

    elif queue <= 5:

        queue_state = 1

    elif queue <= 10:

        queue_state = 2

    else:

        queue_state = 3

    # --------------------------------------------------------
    # Waiting-time state
    # --------------------------------------------------------

    if waiting_time <= 10:

        waiting_state = 0

    elif waiting_time <= 30:

        waiting_state = 1

    elif waiting_time <= 60:

        waiting_state = 2

    else:

        waiting_state = 3

    return (
        queue_state,
        waiting_state,
        phase
    )


# ============================================================
# LOAD Q-TABLE
# ============================================================

def load_q_table():

    if not os.path.exists(Q_TABLE_FILE):

        raise FileNotFoundError(
            f"Q-table not found: {Q_TABLE_FILE}"
        )

    with open(
        Q_TABLE_FILE,
        "rb"
    ) as file:

        q_table = pickle.load(file)

    print(
        f"Q-table loaded successfully."
    )

    print(
        f"Q-table states: {len(q_table)}"
    )

    return q_table


# ============================================================
# CHOOSE BEST ACTION
# ============================================================

def choose_best_action(
    q_table,
    state
):

    # --------------------------------------------------------
    # If state was never learned, keep current phase
    # --------------------------------------------------------

    if state not in q_table:

        return 0

    actions = q_table[state]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not actions:

        return 0

    # --------------------------------------------------------
    # Exploitation only
    #
    # IMPORTANT:
    # During evaluation we do NOT explore randomly.
    # We use the learned Q-table.
    # --------------------------------------------------------

    best_action = max(
        range(len(actions)),
        key=lambda action: actions[action]
    )

    return best_action


# ============================================================
# APPLY ACTION
# ============================================================

def apply_action(
    runner,
    action
):

    if action == 0:

        # Keep current phase
        pass

    elif action == 1:

        current_phase = (
            runner.get_phase()
        )

        next_phase = (
            current_phase + 1
        ) % 2

        runner.set_phase(
            next_phase
        )

    elif action == 2:

        current_phase = (
            runner.get_phase()
        )

        next_phase = (
            current_phase - 1
        ) % 2

        runner.set_phase(
            next_phase
        )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_results(results):

    with open(
        EVALUATION_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Episode",
            "Average_Waiting_Time",
            "Average_Queue",
            "Average_Vehicles",
            "Average_Throughput",
            "Steps"
        ])

        for result in results:

            writer.writerow([
                result["episode"],
                result["waiting_time"],
                result["queue"],
                result["vehicles"],
                result["throughput"],
                result["steps"]
            ])


# ============================================================
# MAIN
# ============================================================

def main():

    clear_screen()

    print("=" * 60)
    print("RL POLICY EVALUATION")
    print("=" * 60)

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print()

    # --------------------------------------------------------
    # LOAD TRAINED Q-TABLE
    # --------------------------------------------------------

    q_table = load_q_table()

    print()

    results = []

    # ========================================================
    # EVALUATION EPISODES
    # ========================================================

    for episode in range(
        1,
        EPISODES + 1
    ):

        print("=" * 60)

        print(
            f"RL EVALUATION EPISODE "
            f"{episode}/{EPISODES}"
        )

        print("=" * 60)

        runner = SUMORunner(
            use_gui=False,
            step_length=1
        )

        waiting_values = []
        queue_values = []
        vehicle_values = []
        throughput_values = []

        steps_completed = 0

        try:

            # ------------------------------------------------
            # START SUMO
            # ------------------------------------------------

            runner.start()

            # ------------------------------------------------
            # INITIAL STATE
            # ------------------------------------------------

            state = get_state(
                runner
            )

            # ------------------------------------------------
            # EVALUATION LOOP
            # ------------------------------------------------

            for step in range(
                STEPS_PER_EPISODE
            ):

                # --------------------------------------------
                # SELECT BEST LEARNED ACTION
                # --------------------------------------------

                action = choose_best_action(
                    q_table,
                    state
                )

                # --------------------------------------------
                # APPLY ACTION
                # --------------------------------------------

                apply_action(
                    runner,
                    action
                )

                # --------------------------------------------
                # ADVANCE SUMO
                # --------------------------------------------

                if not runner.step():

                    break

                steps_completed += 1

                # --------------------------------------------
                # GET TRAFFIC DATA
                # --------------------------------------------

                traffic = (
                    runner.get_traffic_state()
                )

                waiting_values.append(
                    traffic["waiting_time"]
                )

                queue_values.append(
                    traffic["queue"]
                )

                vehicle_values.append(
                    traffic["vehicles"]
                )

                throughput_values.append(
                    traffic["throughput"]
                )

                # --------------------------------------------
                # NEXT STATE
                # --------------------------------------------

                state = get_state(
                    runner
                )

                time.sleep(0.001)

        except Exception as e:

            print()

            print(
                f"Episode {episode} error:"
            )

            print(e)

        finally:

            runner.close()

        # ----------------------------------------------------
        # CALCULATE EPISODE AVERAGES
        # ----------------------------------------------------

        if waiting_values:

            average_waiting = (
                sum(waiting_values)
                / len(waiting_values)
            )

        else:

            average_waiting = 0.0


        if queue_values:

            average_queue = (
                sum(queue_values)
                / len(queue_values)
            )

        else:

            average_queue = 0.0


        if vehicle_values:

            average_vehicles = (
                sum(vehicle_values)
                / len(vehicle_values)
            )

        else:

            average_vehicles = 0.0


        if throughput_values:

            average_throughput = (
                sum(throughput_values)
                / len(throughput_values)
            )

        else:

            average_throughput = 0.0


        result = {

            "episode":
                episode,

            "waiting_time":
                average_waiting,

            "queue":
                average_queue,

            "vehicles":
                average_vehicles,

            "throughput":
                average_throughput,

            "steps":
                steps_completed
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print()

        print("-" * 60)

        print(
            f"Episode {episode} completed"
        )

        print(
            f"Average Waiting Time: "
            f"{average_waiting:.3f}"
        )

        print(
            f"Average Queue: "
            f"{average_queue:.3f}"
        )

        print(
            f"Average Vehicles: "
            f"{average_vehicles:.3f}"
        )

        print(
            f"Average Throughput: "
            f"{average_throughput:.3f}"
        )

        print(
            f"Steps: "
            f"{steps_completed}"
        )

        print("-" * 60)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results(
        results
    )

    # ========================================================
    # OVERALL AVERAGES
    # ========================================================

    if results:

        overall_waiting = sum(
            r["waiting_time"]
            for r in results
        ) / len(results)

        overall_queue = sum(
            r["queue"]
            for r in results
        ) / len(results)

        overall_vehicles = sum(
            r["vehicles"]
            for r in results
        ) / len(results)

        overall_throughput = sum(
            r["throughput"]
            for r in results
        ) / len(results)

    else:

        overall_waiting = 0.0
        overall_queue = 0.0
        overall_vehicles = 0.0
        overall_throughput = 0.0


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("=" * 60)
    print("RL EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"Average Waiting Time: "
        f"{overall_waiting:.3f}"
    )

    print(
        f"Average Queue: "
        f"{overall_queue:.3f}"
    )

    print(
        f"Average Vehicles: "
        f"{overall_vehicles:.3f}"
    )

    print(
        f"Average Throughput: "
        f"{overall_throughput:.3f}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        EVALUATION_FILE
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()