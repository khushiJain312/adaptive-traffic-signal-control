
import os
import csv
import statistics
import traci

from config import SUMO_BINARY, SUMO_CONFIG
from rl_agent import QLearningAgent


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

EPISODES = 5
STEPS = 500

RESULTS_DIR = "results"

Q_TABLE_PATH = os.path.join(
    RESULTS_DIR,
    "q_table.pkl"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "rl_evaluation.csv"
)


# ============================================================
# MUST MATCH TRAINING
# ============================================================

PHASES = 4
MIN_PHASE_TIME = 10
ACTION_COOLDOWN = 5


# ============================================================
# TRAFFIC METRICS
# ============================================================

def get_metrics():

    vehicles = traci.vehicle.getIDList()

    total_waiting = 0.0
    total_queue = 0

    for vehicle_id in vehicles:

        total_waiting += (
            traci.vehicle.getWaitingTime(
                vehicle_id
            )
        )

        speed = traci.vehicle.getSpeed(
            vehicle_id
        )

        if speed < 0.1:
            total_queue += 1

    return (
        len(vehicles),
        total_queue,
        total_waiting
    )


# ============================================================
# START SUMO
# ============================================================

def start_sumo(seed):

    traci.start([
        SUMO_BINARY,
        "-c",
        SUMO_CONFIG,
        "--start",
        "--quit-on-end",
        "--seed",
        str(seed)
    ])


# ============================================================
# REWARD
# MUST MATCH training.py
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
        previous_traffic["throughput"]
    )

    current_throughput = (
        current_traffic["throughput"]
    )

    # --------------------------------------------------------
    # Traffic changes
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
    # Reward
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
    # Bound reward
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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RL TRAFFIC SIGNAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD Q-TABLE
    # --------------------------------------------------------

    print(
        "Loading trained Q-table..."
    )

    agent = QLearningAgent(
        learning_rate=0.10,
        discount_factor=0.95,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0
    )

    agent.load(
        Q_TABLE_PATH
    )

    print(
        "Q-table loaded successfully."
    )

    print(
        "Q-table states:",
        agent.get_q_table_size()
    )

    episode_results = []


    # ========================================================
    # EVALUATION EPISODES
    # ========================================================

    for episode in range(
        1,
        EPISODES + 1
    ):

        print()
        print("=" * 60)

        print(
            f"EVALUATION EPISODE "
            f"{episode}/{EPISODES}"
        )

        print("=" * 60)

        # Different seed for every evaluation episode
        seed = 1000 + episode

        print(
            f"SUMO random seed: {seed}"
        )

        start_sumo(seed)

        traffic_lights = (
            traci.trafficlight.getIDList()
        )

        if not traffic_lights:

            print(
                "ERROR: No traffic light found."
            )

            traci.close()

            return

        traffic_light = traffic_lights[0]

        total_reward = 0.0
        total_waiting = 0.0
        total_queue = 0
        total_vehicles = 0

        phase_timer = 0
        cooldown_timer = 0

        try:

            # ------------------------------------------------
            # INITIAL TRAFFIC
            # ------------------------------------------------

            vehicles, queue, waiting = (
                get_metrics()
            )

            phase = (
                traci.trafficlight.getPhase(
                    traffic_light
                )
            )

            throughput = (
                traci.simulation.getArrivedNumber()
            )

            previous_traffic = {

                "vehicles":
                    vehicles,

                "queue":
                    queue,

                "waiting_time":
                    waiting,

                "throughput":
                    throughput,

                "phase":
                    phase
            }


            # ------------------------------------------------
            # INITIAL STATE
            # ------------------------------------------------

            state = agent.get_state(
                queue,
                waiting,
                phase
            )


            # =================================================
            # SIMULATION LOOP
            # =================================================

            for step in range(
                1,
                STEPS + 1
            ):

                # --------------------------------------------
                # UPDATE TIMERS
                # --------------------------------------------

                phase_timer += 1

                if cooldown_timer > 0:

                    cooldown_timer -= 1


                # --------------------------------------------
                # GREEDY ACTION
                # --------------------------------------------

                if state in agent.q_table:

                    action = agent.get_best_action(
                        state
                    )

                else:

                    action = 0


                # --------------------------------------------
                # CHECK WHETHER CHANGE IS ALLOWED
                # --------------------------------------------

                can_change = (
                    phase_timer >= MIN_PHASE_TIME
                    and cooldown_timer == 0
                )


                # --------------------------------------------
                # APPLY ACTION
                # --------------------------------------------

                if (
                    action == 1
                    and can_change
                ):

                    current_phase = (
                        traci.trafficlight.getPhase(
                            traffic_light
                        )
                    )

                    next_phase = (
                        current_phase + 1
                    ) % PHASES

                    traci.trafficlight.setPhase(
                        traffic_light,
                        next_phase
                    )

                    phase = next_phase

                    phase_timer = 0

                    cooldown_timer = (
                        ACTION_COOLDOWN
                    )

                else:

                    phase = (
                        traci.trafficlight.getPhase(
                            traffic_light
                        )
                    )


                # --------------------------------------------
                # ADVANCE SUMO
                # --------------------------------------------

                traci.simulationStep()


                # --------------------------------------------
                # UPDATED TRAFFIC
                # --------------------------------------------

                vehicles, queue, waiting = (
                    get_metrics()
                )

                throughput = (
                    traci.simulation.getArrivedNumber()
                )

                current_traffic = {

                    "vehicles":
                        vehicles,

                    "queue":
                        queue,

                    "waiting_time":
                        waiting,

                    "throughput":
                        throughput,

                    "phase":
                        phase
                }


                # --------------------------------------------
                # CALCULATE REWARD
                # --------------------------------------------

                reward = calculate_reward(
                    previous_traffic,
                    current_traffic
                )

                total_reward += reward


                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                total_waiting += waiting
                total_queue += queue
                total_vehicles += vehicles


                # --------------------------------------------
                # DISPLAY
                # --------------------------------------------

                if step % 50 == 0:

                    print(
                        f"Step: {step:4d} | "
                        f"Vehicles: {vehicles:3d} | "
                        f"Queue: {queue:3d} | "
                        f"Waiting: {waiting:8.2f} | "
                        f"Phase: {phase:2d} | "
                        f"Action: "
                        f"{'CHANGE' if action == 1 else 'KEEP':6s} | "
                        f"Reward: {reward:8.2f}"
                    )


                # --------------------------------------------
                # NEXT STATE
                # --------------------------------------------

                state = agent.get_state(
                    queue,
                    waiting,
                    phase
                )

                previous_traffic = (
                    current_traffic
                )


            # =================================================
            # EPISODE SUMMARY
            # =================================================

            avg_waiting = (
                total_waiting / STEPS
            )

            avg_queue = (
                total_queue / STEPS
            )

            avg_vehicles = (
                total_vehicles / STEPS
            )

            arrived = (
                traci.simulation.getArrivedNumber()
            )

            print()
            print("Episode Summary")
            print("-" * 60)

            print(
                f"Average Waiting Time : "
                f"{avg_waiting:.3f}"
            )

            print(
                f"Average Queue        : "
                f"{avg_queue:.3f}"
            )

            print(
                f"Average Vehicles     : "
                f"{avg_vehicles:.3f}"
            )

            print(
                f"Throughput           : "
                f"{arrived}"
            )

            print(
                f"Total Reward         : "
                f"{total_reward:.3f}"
            )

            episode_results.append({

                "episode":
                    episode,

                "seed":
                    seed,

                "avg_waiting":
                    avg_waiting,

                "avg_queue":
                    avg_queue,

                "avg_vehicles":
                    avg_vehicles,

                "throughput":
                    arrived,

                "total_reward":
                    total_reward
            })


        finally:

            traci.close()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL RL EVALUATION")
    print("=" * 60)

    waiting_values = [
        x["avg_waiting"]
        for x in episode_results
    ]

    queue_values = [
        x["avg_queue"]
        for x in episode_results
    ]

    vehicle_values = [
        x["avg_vehicles"]
        for x in episode_results
    ]

    throughput_values = [
        x["throughput"]
        for x in episode_results
    ]

    reward_values = [
        x["total_reward"]
        for x in episode_results
    ]


    # --------------------------------------------------------
    # MEANS
    # --------------------------------------------------------

    avg_waiting = statistics.mean(
        waiting_values
    )

    avg_queue = statistics.mean(
        queue_values
    )

    avg_vehicles = statistics.mean(
        vehicle_values
    )

    avg_throughput = statistics.mean(
        throughput_values
    )

    avg_reward = statistics.mean(
        reward_values
    )


    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    std_waiting = (
        statistics.stdev(waiting_values)
        if len(waiting_values) > 1
        else 0.0
    )

    std_queue = (
        statistics.stdev(queue_values)
        if len(queue_values) > 1
        else 0.0
    )

    std_vehicles = (
        statistics.stdev(vehicle_values)
        if len(vehicle_values) > 1
        else 0.0
    )

    std_throughput = (
        statistics.stdev(throughput_values)
        if len(throughput_values) > 1
        else 0.0
    )

    std_reward = (
        statistics.stdev(reward_values)
        if len(reward_values) > 1
        else 0.0
    )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print(
        f"Average Waiting Time : "
        f"{avg_waiting:.3f} "
        f"+/- {std_waiting:.3f}"
    )

    print(
        f"Average Queue        : "
        f"{avg_queue:.3f} "
        f"+/- {std_queue:.3f}"
    )

    print(
        f"Average Vehicles     : "
        f"{avg_vehicles:.3f} "
        f"+/- {std_vehicles:.3f}"
    )

    print(
        f"Average Throughput   : "
        f"{avg_throughput:.3f} "
        f"+/- {std_throughput:.3f}"
    )

    print(
        f"Average Reward       : "
        f"{avg_reward:.3f} "
        f"+/- {std_reward:.3f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Episode",
            "Seed",
            "Average_Waiting_Time",
            "Average_Queue",
            "Average_Vehicles",
            "Throughput",
            "Total_Reward"
        ])

        for result in episode_results:

            writer.writerow([
                result["episode"],
                result["seed"],
                f"{result['avg_waiting']:.3f}",
                f"{result['avg_queue']:.3f}",
                f"{result['avg_vehicles']:.3f}",
                result["throughput"],
                f"{result['total_reward']:.3f}"
            ])


        # ----------------------------------------------------
        # SUMMARY ROW
        # ----------------------------------------------------

        writer.writerow([])

        writer.writerow([
            "MEAN",
            "",
            f"{avg_waiting:.3f}",
            f"{avg_queue:.3f}",
            f"{avg_vehicles:.3f}",
            f"{avg_throughput:.3f}",
            f"{avg_reward:.3f}"
        ])

        writer.writerow([
            "STD",
            "",
            f"{std_waiting:.3f}",
            f"{std_queue:.3f}",
            f"{std_vehicles:.3f}",
            f"{std_throughput:.3f}",
            f"{std_reward:.3f}"
        ])


    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
