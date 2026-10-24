
import os
import csv
import pickle
import time

from sumo_runner import SUMORunner
from rl_agent import QLearningAgent


# ============================================================
# COMPARISON CONFIGURATION
# ============================================================

EPISODES = 5
STEPS_PER_EPISODE = 500

RESULTS_DIR = "results"

Q_TABLE_FILE = os.path.join(
    RESULTS_DIR,
    "q_table.pkl"
)

EVOLUTION_POLICY_FILE = os.path.join(
    RESULTS_DIR,
    "best_evolution_policy.csv"
)

COMPARISON_FILE = os.path.join(
    RESULTS_DIR,
    "comparison_results.csv"
)

# Baseline fixed-time signal
BASELINE_PHASE_1 = 30
BASELINE_PHASE_2 = 30

PHASES = 2


# ============================================================
# LOAD EVOLUTIONARY POLICY
# ============================================================

def load_evolution_policy():

    policy = []

    with open(
        EVOLUTION_POLICY_FILE,
        "r"
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) >= 2:

                try:

                    phase = int(row[0])
                    duration = int(row[1])

                    if 0 <= phase < PHASES:

                        while len(policy) <= phase:
                            policy.append(None)

                        policy[phase] = duration

                except ValueError:

                    continue

    if (
        len(policy) != PHASES
        or any(value is None for value in policy)
    ):

        raise ValueError(
            "Could not load a valid evolutionary policy."
        )

    return policy


# ============================================================
# LOAD Q-TABLE
# ============================================================

def load_q_table():

    agent = QLearningAgent()

    agent.load(
        Q_TABLE_FILE
    )

    return agent


# ============================================================
# GET TRAFFIC METRICS
# ============================================================

def get_metrics(runner):

    traffic = (
        runner.get_traffic_state()
    )

    return (
        traffic["vehicles"],
        traffic["queue"],
        traffic["waiting_time"],
        traffic["throughput"]
    )


# ============================================================
# GET BEST Q-LEARNING ACTION
# ============================================================

def get_q_learning_action(
    agent,
    state
):

    # --------------------------------------------------------
    # Get Q-values for current state
    # --------------------------------------------------------

    q_values = agent.q_table.get(
        state,
        None
    )

    # Unknown state
    if q_values is None:

        return 0

    # --------------------------------------------------------
    # Dictionary format
    #
    # {
    #     0: value,
    #     1: value
    # }
    # --------------------------------------------------------

    if isinstance(
        q_values,
        dict
    ):

        if not q_values:

            return 0

        return max(
            q_values,
            key=q_values.get
        )

    # --------------------------------------------------------
    # List / tuple format
    #
    # [value_for_action_0,
    #  value_for_action_1]
    # --------------------------------------------------------

    if isinstance(
        q_values,
        (list, tuple)
    ):

        if len(q_values) == 0:

            return 0

        return max(
            range(len(q_values)),
            key=lambda i: q_values[i]
        )

    # --------------------------------------------------------
    # Unexpected format
    # --------------------------------------------------------

    print(
        f"Warning: Unsupported Q-value format "
        f"for state {state}: "
        f"{type(q_values).__name__}"
    )

    return 0


# ============================================================
# RUN BASELINE
# ============================================================

def evaluate_baseline():

    print()
    print("=" * 60)
    print("EVALUATING BASELINE")
    print("=" * 60)

    all_results = []

    for episode in range(
        1,
        EPISODES + 1
    ):

        runner = SUMORunner(
            use_gui=False,
            step_length=1
        )

        total_waiting = 0.0
        total_queue = 0.0
        total_vehicles = 0.0
        total_throughput = 0.0

        steps_completed = 0

        try:

            runner.start()

            phase = runner.get_phase()

            phase_timer = 0

            for step in range(
                STEPS_PER_EPISODE
            ):

                if phase == 0:

                    phase_duration = (
                        BASELINE_PHASE_1
                    )

                else:

                    phase_duration = (
                        BASELINE_PHASE_2
                    )

                phase_timer += 1

                if phase_timer >= phase_duration:

                    phase = (
                        phase + 1
                    ) % PHASES

                    runner.set_phase(
                        phase
                    )

                    phase_timer = 0

                if not runner.step():

                    break

                steps_completed += 1

                (
                    vehicles,
                    queue,
                    waiting,
                    throughput
                ) = get_metrics(runner)

                total_waiting += waiting
                total_queue += queue
                total_vehicles += vehicles
                total_throughput += throughput

                time.sleep(0.001)

        except Exception as e:

            print(
                f"Baseline episode {episode} error: {e}"
            )

            continue

        finally:

            runner.close()

        if steps_completed == 0:

            continue

        result = {

            "controller":
                "Baseline",

            "episode":
                episode,

            "avg_waiting":
                total_waiting / steps_completed,

            "avg_queue":
                total_queue / steps_completed,

            "avg_vehicles":
                total_vehicles / steps_completed,

            "throughput":
                total_throughput / steps_completed
        }

        all_results.append(
            result
        )

        print(
            f"Episode {episode} | "
            f"Waiting: {result['avg_waiting']:.3f} | "
            f"Queue: {result['avg_queue']:.3f} | "
            f"Vehicles: {result['avg_vehicles']:.3f} | "
            f"Throughput: {result['throughput']:.3f}"
        )

    return all_results


# ============================================================
# RUN Q-LEARNING
# ============================================================

def evaluate_q_learning(agent):

    print()
    print("=" * 60)
    print("EVALUATING Q-LEARNING")
    print("=" * 60)

    all_results = []

    for episode in range(
        1,
        EPISODES + 1
    ):

        runner = SUMORunner(
            use_gui=False,
            step_length=1
        )

        total_waiting = 0.0
        total_queue = 0.0
        total_vehicles = 0.0
        total_throughput = 0.0

        steps_completed = 0

        try:

            runner.start()

            # ------------------------------------------------
            # INITIAL STATE
            # ------------------------------------------------

            traffic = (
                runner.get_traffic_state()
            )

            state = agent.get_state(
                traffic["queue"],
                traffic["waiting_time"],
                traffic["phase"]
            )

            # ------------------------------------------------
            # SIMULATION
            # ------------------------------------------------

            for step in range(
                STEPS_PER_EPISODE
            ):

                # --------------------------------------------
                # GREEDY ACTION
                # --------------------------------------------

                action = get_q_learning_action(
                    agent,
                    state
                )

                # --------------------------------------------
                # APPLY ACTION
                # --------------------------------------------

                if action == 1:

                    current_phase = (
                        runner.get_phase()
                    )

                    next_phase = (
                        current_phase + 1
                    ) % PHASES

                    runner.set_phase(
                        next_phase
                    )

                # --------------------------------------------
                # STEP SUMO
                # --------------------------------------------

                if not runner.step():

                    break

                steps_completed += 1

                # --------------------------------------------
                # NEXT STATE
                # --------------------------------------------

                traffic = (
                    runner.get_traffic_state()
                )

                next_state = agent.get_state(
                    traffic["queue"],
                    traffic["waiting_time"],
                    traffic["phase"]
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                vehicles = (
                    traffic["vehicles"]
                )

                queue = (
                    traffic["queue"]
                )

                waiting = (
                    traffic["waiting_time"]
                )

                throughput = (
                    traffic["throughput"]
                )

                total_waiting += waiting
                total_queue += queue
                total_vehicles += vehicles
                total_throughput += throughput

                state = next_state

                time.sleep(0.001)

        except Exception as e:

            print(
                f"Q-Learning episode {episode} error: {e}"
            )

            continue

        finally:

            runner.close()

        if steps_completed == 0:

            continue

        result = {

            "controller":
                "Q-Learning",

            "episode":
                episode,

            "avg_waiting":
                total_waiting / steps_completed,

            "avg_queue":
                total_queue / steps_completed,

            "avg_vehicles":
                total_vehicles / steps_completed,

            "throughput":
                total_throughput / steps_completed
        }

        all_results.append(
            result
        )

        print(
            f"Episode {episode} | "
            f"Waiting: {result['avg_waiting']:.3f} | "
            f"Queue: {result['avg_queue']:.3f} | "
            f"Vehicles: {result['avg_vehicles']:.3f} | "
            f"Throughput: {result['throughput']:.3f}"
        )

    return all_results


# ============================================================
# RUN EVOLUTIONARY POLICY
# ============================================================

def evaluate_evolutionary(policy):

    print()
    print("=" * 60)
    print("EVALUATING EVOLUTIONARY ALGORITHM")
    print("=" * 60)

    print(
        f"Best policy: {policy}"
    )

    all_results = []

    for episode in range(
        1,
        EPISODES + 1
    ):

        runner = SUMORunner(
            use_gui=False,
            step_length=1
        )

        total_waiting = 0.0
        total_queue = 0.0
        total_vehicles = 0.0
        total_throughput = 0.0

        steps_completed = 0

        try:

            runner.start()

            phase = runner.get_phase()

            phase_timer = 0

            for step in range(
                STEPS_PER_EPISODE
            ):

                phase_duration = (
                    policy[phase]
                )

                phase_timer += 1

                if phase_timer >= phase_duration:

                    phase = (
                        phase + 1
                    ) % PHASES

                    runner.set_phase(
                        phase
                    )

                    phase_timer = 0

                if not runner.step():

                    break

                steps_completed += 1

                (
                    vehicles,
                    queue,
                    waiting,
                    throughput
                ) = get_metrics(runner)

                total_waiting += waiting
                total_queue += queue
                total_vehicles += vehicles
                total_throughput += throughput

                time.sleep(0.001)

        except Exception as e:

            print(
                f"Evolutionary episode {episode} error: {e}"
            )

            continue

        finally:

            runner.close()

        if steps_completed == 0:

            continue

        result = {

            "controller":
                "Evolutionary",

            "episode":
                episode,

            "avg_waiting":
                total_waiting / steps_completed,

            "avg_queue":
                total_queue / steps_completed,

            "avg_vehicles":
                total_vehicles / steps_completed,

            "throughput":
                total_throughput / steps_completed
        }

        all_results.append(
            result
        )

        print(
            f"Episode {episode} | "
            f"Waiting: {result['avg_waiting']:.3f} | "
            f"Queue: {result['avg_queue']:.3f} | "
            f"Vehicles: {result['avg_vehicles']:.3f} | "
            f"Throughput: {result['throughput']:.3f}"
        )

    return all_results


# ============================================================
# CALCULATE FINAL AVERAGES
# ============================================================

def calculate_average(results):

    if not results:

        return {
            "avg_waiting": None,
            "avg_queue": None,
            "avg_vehicles": None,
            "throughput": None
        }

    return {

        "avg_waiting":
            sum(
                x["avg_waiting"]
                for x in results
            ) / len(results),

        "avg_queue":
            sum(
                x["avg_queue"]
                for x in results
            ) / len(results),

        "avg_vehicles":
            sum(
                x["avg_vehicles"]
                for x in results
            ) / len(results),

        "throughput":
            sum(
                x["throughput"]
                for x in results
            ) / len(results)
    }


# ============================================================
# SAVE COMPARISON
# ============================================================

def save_comparison(all_results):

    with open(
        COMPARISON_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Controller",
            "Average_Waiting_Time",
            "Average_Queue",
            "Average_Vehicles",
            "Average_Throughput"
        ])

        for controller, results in all_results.items():

            average = calculate_average(
                results
            )

            if average["avg_waiting"] is None:

                writer.writerow([
                    controller,
                    "ERROR",
                    "ERROR",
                    "ERROR",
                    "ERROR"
                ])

            else:

                writer.writerow([
                    controller,
                    f"{average['avg_waiting']:.3f}",
                    f"{average['avg_queue']:.3f}",
                    f"{average['avg_vehicles']:.3f}",
                    f"{average['throughput']:.3f}"
                ])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAFFIC SIGNAL CONTROLLER COMPARISON")
    print("=" * 60)

    print(
        f"Episodes per controller: {EPISODES}"
    )

    print(
        f"Steps per episode: {STEPS_PER_EPISODE}"
    )

    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(
        Q_TABLE_FILE
    ):

        print(
            "ERROR: q_table.pkl not found."
        )

        print(
            "Run training.py first."
        )

        return

    if not os.path.exists(
        EVOLUTION_POLICY_FILE
    ):

        print(
            "ERROR: best_evolution_policy.csv not found."
        )

        print(
            "Run evolution.py first."
        )

        return

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print(
        "Loading Q-learning model..."
    )

    agent = load_q_table()

    print(
        f"Q-table states: "
        f"{agent.get_q_table_size()}"
    )

    print()

    print(
        "Loading evolutionary policy..."
    )

    evolution_policy = (
        load_evolution_policy()
    )

    print(
        f"Evolutionary policy: "
        f"{evolution_policy}"
    )

    # ========================================================
    # EVALUATE ALL CONTROLLERS
    # ========================================================

    baseline_results = (
        evaluate_baseline()
    )

    q_learning_results = (
        evaluate_q_learning(
            agent
        )
    )

    evolutionary_results = (
        evaluate_evolutionary(
            evolution_policy
        )
    )

    all_results = {

        "Baseline":
            baseline_results,

        "Q-Learning":
            q_learning_results,

        "Evolutionary":
            evolutionary_results
    }

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        f"{'Controller':<18}"
        f"{'Waiting':>12}"
        f"{'Queue':>12}"
        f"{'Vehicles':>12}"
        f"{'Throughput':>14}"
    )

    print("-" * 68)

    for controller, results in all_results.items():

        average = calculate_average(
            results
        )

        if average["avg_waiting"] is None:

            print(
                f"{controller:<18}"
                f"{'ERROR':>12}"
                f"{'ERROR':>12}"
                f"{'ERROR':>12}"
                f"{'ERROR':>14}"
            )

        else:

            print(
                f"{controller:<18}"
                f"{average['avg_waiting']:>12.3f}"
                f"{average['avg_queue']:>12.3f}"
                f"{average['avg_vehicles']:>12.3f}"
                f"{average['throughput']:>14.3f}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    save_comparison(
        all_results
    )

    print()
    print(
        "Comparison results saved to:"
    )

    print(
        COMPARISON_FILE
    )

    print()
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
