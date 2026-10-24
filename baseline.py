
import os
import pandas as pd

from sumo_runner import SUMORunner


# ============================================================
# BASELINE CONFIGURATION
# ============================================================

EPISODES = 30
STEPS_PER_EPISODE = 500


# ============================================================
# RUN ONE BASELINE EPISODE
# ============================================================

def run_baseline_episode(episode):

    print("=" * 60)
    print(
        f"FIXED-TIME BASELINE EPISODE {episode}"
    )
    print("=" * 60)

    runner = SUMORunner(
        use_gui=False,
        step_length=1
    )

    runner.start()

    total_queue = 0
    total_waiting = 0
    total_vehicles = 0
    total_throughput = 0

    steps_completed = 0

    try:

        for step in range(
            STEPS_PER_EPISODE
        ):

            # Advance SUMO by one step
            if not runner.step():
                break

            # Collect metrics
            total_queue += (
                runner.get_queue_length()
            )

            total_waiting += (
                runner.get_waiting_time()
            )

            total_vehicles += (
                runner.get_vehicle_count()
            )

            # get_throughput() returns
            # vehicles that arrived during
            # the current simulation step
            total_throughput += (
                runner.get_throughput()
            )

            steps_completed += 1


        if steps_completed == 0:

            steps_completed = 1


        # ====================================================
        # CALCULATE AVERAGES
        # ====================================================

        average_queue = (
            total_queue /
            steps_completed
        )

        average_waiting = (
            total_waiting /
            steps_completed
        )

        average_vehicles = (
            total_vehicles /
            steps_completed
        )

        # Throughput is the total number
        # of vehicles that completed their
        # trips during the episode.
        average_throughput = (
            total_throughput /
            steps_completed
        )


        # ====================================================
        # EPISODE RESULTS
        # ====================================================

        result = {

            "Episode":
                episode,

            "Average_Waiting_Time":
                average_waiting,

            "Average_Queue":
                average_queue,

            "Average_Vehicles":
                average_vehicles,

            "Average_Throughput":
                average_throughput,

            "Steps":
                steps_completed
        }


        print(
            f"Average Waiting Time: "
            f"{average_waiting:.2f}"
        )

        print(
            f"Average Queue: "
            f"{average_queue:.2f}"
        )

        print(
            f"Average Vehicles: "
            f"{average_vehicles:.2f}"
        )

        print(
            f"Average Throughput: "
            f"{average_throughput:.2f}"
        )

        print(
            f"Steps: "
            f"{steps_completed}"
        )

        return result


    finally:

        runner.close()


# ============================================================
# MAIN
# ============================================================

def main():

    results = []


    for episode in range(
        1,
        EPISODES + 1
    ):

        result = run_baseline_episode(
            episode
        )

        results.append(result)


    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )


    # ========================================================
    # CREATE RESULTS DIRECTORY
    # ========================================================

    results_dir = "results"

    os.makedirs(
        results_dir,
        exist_ok=True
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    output_file = os.path.join(
        results_dir,
        "baseline_results.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "FIXED-TIME BASELINE COMPLETE"
    )
    print("=" * 60)

    print()

    print(df.to_string(index=False))

    print()

    print(
        "Results saved to:"
    )

    print(
        os.path.abspath(
            output_file
        )
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
