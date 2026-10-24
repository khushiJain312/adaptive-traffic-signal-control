import os
import random
import csv
import time

from sumo_runner import SUMORunner


# ============================================================
# EVOLUTIONARY ALGORITHM CONFIGURATION
# ============================================================

POPULATION_SIZE = 10
GENERATIONS = 20

STEPS_PER_EPISODE = 500

MUTATION_RATE = 0.20
ELITE_COUNT = 2

# Current project uses 2 traffic-light phases
PHASES = 2

# Minimum / maximum green duration
MIN_GREEN = 10
MAX_GREEN = 60

RESULTS_DIR = "results"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "evolution_results.csv"
)

BEST_POLICY_FILE = os.path.join(
    RESULTS_DIR,
    "best_evolution_policy.csv"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CREATE RANDOM POLICY
# ============================================================

def create_policy():

    return [
        random.randint(
            MIN_GREEN,
            MAX_GREEN
        )
        for _ in range(PHASES)
    ]


# ============================================================
# CREATE INITIAL POPULATION
# ============================================================

def create_population():

    return [
        create_policy()
        for _ in range(POPULATION_SIZE)
    ]


# ============================================================
# FITNESS FUNCTION
# ============================================================

def calculate_fitness(
    avg_waiting,
    avg_queue,
    throughput
):
    """
    Higher fitness = better traffic performance.

    Lower waiting time  -> better
    Lower queue         -> better
    Higher throughput   -> better
    """

    fitness = (
        -avg_waiting
        - (avg_queue * 2.0)
        + (throughput * 0.5)
    )

    return fitness


# ============================================================
# EVALUATE POLICY
# ============================================================

def evaluate_policy(
    policy,
    generation,
    individual
):

    runner = SUMORunner(
        use_gui=False,
        step_length=1
    )

    total_waiting = 0.0
    total_queue = 0
    total_vehicles = 0

    steps_completed = 0

    # --------------------------------------------------------
    # Throughput is cumulative in SUMO.
    # We therefore only take the FINAL value.
    # --------------------------------------------------------

    final_throughput = 0

    try:

        # ----------------------------------------------------
        # START SUMO
        # ----------------------------------------------------

        runner.start()

        # ----------------------------------------------------
        # INITIAL PHASE
        # ----------------------------------------------------

        current_phase = int(
            runner.get_phase()
        )

        # Make sure the phase is valid
        current_phase %= PHASES

        phase_timer = 0

        # ----------------------------------------------------
        # SIMULATION LOOP
        # ----------------------------------------------------

        for step in range(
            STEPS_PER_EPISODE
        ):

            # -----------------------------------------------
            # CURRENT PHASE DURATION
            # -----------------------------------------------

            phase_duration = policy[
                current_phase
            ]

            phase_timer += 1

            # -----------------------------------------------
            # CHANGE PHASE
            # -----------------------------------------------

            if phase_timer >= phase_duration:

                current_phase = (
                    current_phase + 1
                ) % PHASES

                runner.set_phase(
                    current_phase
                )

                phase_timer = 0

            # -----------------------------------------------
            # ADVANCE SUMO
            # -----------------------------------------------

            if not runner.step():

                break

            steps_completed += 1

            # -----------------------------------------------
            # GET TRAFFIC METRICS
            # -----------------------------------------------

            traffic = (
                runner.get_traffic_state()
            )

            total_waiting += float(
                traffic["waiting_time"]
            )

            total_queue += int(
                traffic["queue"]
            )

            total_vehicles += int(
                traffic["vehicles"]
            )

            # -----------------------------------------------
            # IMPORTANT:
            # Throughput is cumulative.
            # Do NOT add it every step.
            # -----------------------------------------------

            final_throughput = int(
                traffic["throughput"]
            )

            time.sleep(0.001)

    except Exception as e:

        print()
        print(
            f"Policy evaluation error: {e}"
        )

        return float("-inf")

    finally:

        runner.close()

    # ========================================================
    # CHECK SIMULATION
    # ========================================================

    if steps_completed == 0:

        return float("-inf")

    # ========================================================
    # AVERAGE METRICS
    # ========================================================

    avg_waiting = (
        total_waiting /
        steps_completed
    )

    avg_queue = (
        total_queue /
        steps_completed
    )

    avg_vehicles = (
        total_vehicles /
        steps_completed
    )

    # ========================================================
    # FITNESS
    # ========================================================

    fitness = calculate_fitness(
        avg_waiting,
        avg_queue,
        final_throughput
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"Generation {generation:2d} | "
        f"Individual {individual:2d} | "
        f"Policy {policy} | "
        f"Waiting {avg_waiting:8.2f} | "
        f"Queue {avg_queue:6.2f} | "
        f"Vehicles {avg_vehicles:6.2f} | "
        f"Throughput {final_throughput:4d} | "
        f"Fitness {fitness:10.2f}"
    )

    return {
        "fitness": fitness,
        "waiting": avg_waiting,
        "queue": avg_queue,
        "vehicles": avg_vehicles,
        "throughput": final_throughput
    }


# ============================================================
# SELECTION
# ============================================================

def select_parent(
    population,
    fitnesses
):

    tournament_size = 3

    candidates = random.sample(
        range(len(population)),
        min(
            tournament_size,
            len(population)
        )
    )

    winner = max(
        candidates,
        key=lambda index: fitnesses[index]
    )

    return population[winner]


# ============================================================
# CROSSOVER
# ============================================================

def crossover(
    parent1,
    parent2
):

    child = []

    for i in range(PHASES):

        if random.random() < 0.5:

            child.append(
                parent1[i]
            )

        else:

            child.append(
                parent2[i]
            )

    return child


# ============================================================
# MUTATION
# ============================================================

def mutate(policy):

    mutated_policy = policy.copy()

    for i in range(PHASES):

        if random.random() < MUTATION_RATE:

            change = random.randint(
                -10,
                10
            )

            mutated_policy[i] = max(
                MIN_GREEN,
                min(
                    MAX_GREEN,
                    mutated_policy[i] + change
                )
            )

    return mutated_policy


# ============================================================
# CREATE NEXT GENERATION
# ============================================================

def create_next_generation(
    population,
    fitnesses
):

    # --------------------------------------------------------
    # RANK POPULATION
    # --------------------------------------------------------

    ranked = sorted(
        zip(
            population,
            fitnesses
        ),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # ELITISM
    # --------------------------------------------------------

    next_population = [
        policy.copy()
        for policy, fitness
        in ranked[:ELITE_COUNT]
    ]

    # --------------------------------------------------------
    # GENERATE CHILDREN
    # --------------------------------------------------------

    while len(next_population) < POPULATION_SIZE:

        parent1 = select_parent(
            population,
            fitnesses
        )

        parent2 = select_parent(
            population,
            fitnesses
        )

        child = crossover(
            parent1,
            parent2
        )

        child = mutate(
            child
        )

        next_population.append(
            child
        )

    return next_population


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
            "Generation",
            "Best_Fitness",
            "Average_Fitness",
            "Best_Waiting_Time",
            "Best_Queue",
            "Best_Vehicles",
            "Best_Throughput",
            "Phase_1_Duration",
            "Phase_2_Duration"
        ])

        for result in results:

            writer.writerow([
                result["generation"],
                f"{result['best_fitness']:.3f}",
                f"{result['average_fitness']:.3f}",
                f"{result['best_waiting']:.3f}",
                f"{result['best_queue']:.3f}",
                f"{result['best_vehicles']:.3f}",
                result["best_throughput"],
                result["phase_1"],
                result["phase_2"]
            ])


# ============================================================
# SAVE BEST POLICY
# ============================================================

def save_best_policy(
    policy,
    fitness,
    waiting,
    queue,
    vehicles,
    throughput
):

    with open(
        BEST_POLICY_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Phase",
            "Duration"
        ])

        for phase, duration in enumerate(
            policy
        ):

            writer.writerow([
                phase,
                duration
            ])

        writer.writerow([])

        writer.writerow([
            "Best_Fitness",
            fitness
        ])

        writer.writerow([
            "Average_Waiting_Time",
            waiting
        ])

        writer.writerow([
            "Average_Queue",
            queue
        ])

        writer.writerow([
            "Average_Vehicles",
            vehicles
        ])

        writer.writerow([
            "Throughput",
            throughput
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("EVOLUTIONARY TRAFFIC SIGNAL OPTIMIZATION")
    print("=" * 60)

    print(
        f"Population size : {POPULATION_SIZE}"
    )

    print(
        f"Generations     : {GENERATIONS}"
    )

    print(
        f"Steps/episode   : {STEPS_PER_EPISODE}"
    )

    print(
        f"Mutation rate   : {MUTATION_RATE}"
    )

    print(
        f"Phase count     : {PHASES}"
    )

    print()

    # ========================================================
    # INITIAL POPULATION
    # ========================================================

    population = create_population()

    results = []

    global_best_policy = None
    global_best_fitness = float("-inf")

    global_best_metrics = {
        "waiting": 0.0,
        "queue": 0.0,
        "vehicles": 0.0,
        "throughput": 0
    }

    # ========================================================
    # EVOLUTION LOOP
    # ========================================================

    for generation in range(
        1,
        GENERATIONS + 1
    ):

        print()
        print("=" * 60)
        print(
            f"GENERATION {generation}/{GENERATIONS}"
        )
        print("=" * 60)

        fitnesses = []
        evaluation_results = []

        # ----------------------------------------------------
        # EVALUATE EVERY POLICY
        # ----------------------------------------------------

        for individual, policy in enumerate(
            population,
            start=1
        ):

            evaluation = evaluate_policy(
                policy,
                generation,
                individual
            )

            evaluation_results.append(
                evaluation
            )

            fitnesses.append(
                evaluation["fitness"]
            )

        # ----------------------------------------------------
        # FIND BEST POLICY
        # ----------------------------------------------------

        best_index = max(
            range(len(fitnesses)),
            key=lambda i: fitnesses[i]
        )

        best_policy = (
            population[best_index]
        )

        best_result = (
            evaluation_results[best_index]
        )

        best_fitness = (
            best_result["fitness"]
        )

        average_fitness = (
            sum(fitnesses) /
            len(fitnesses)
        )

        # ----------------------------------------------------
        # GLOBAL BEST
        # ----------------------------------------------------

        if best_fitness > global_best_fitness:

            global_best_fitness = (
                best_fitness
            )

            global_best_policy = (
                best_policy.copy()
            )

            global_best_metrics = {
                "waiting":
                    best_result["waiting"],

                "queue":
                    best_result["queue"],

                "vehicles":
                    best_result["vehicles"],

                "throughput":
                    best_result["throughput"]
            }

        # ----------------------------------------------------
        # DISPLAY SUMMARY
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print(
            f"Generation {generation} Summary"
        )
        print("-" * 60)

        print(
            f"Best Fitness     : "
            f"{best_fitness:.3f}"
        )

        print(
            f"Average Fitness  : "
            f"{average_fitness:.3f}"
        )

        print(
            f"Best Policy      : "
            f"{best_policy}"
        )

        print(
            f"Waiting Time     : "
            f"{best_result['waiting']:.3f}"
        )

        print(
            f"Queue            : "
            f"{best_result['queue']:.3f}"
        )

        print(
            f"Vehicles         : "
            f"{best_result['vehicles']:.3f}"
        )

        print(
            f"Throughput       : "
            f"{best_result['throughput']}"
        )

        print(
            f"Global Best      : "
            f"{global_best_policy}"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # SAVE GENERATION RESULT
        # ----------------------------------------------------

        results.append({

            "generation":
                generation,

            "best_fitness":
                best_fitness,

            "average_fitness":
                average_fitness,

            "best_waiting":
                best_result["waiting"],

            "best_queue":
                best_result["queue"],

            "best_vehicles":
                best_result["vehicles"],

            "best_throughput":
                best_result["throughput"],

            "phase_1":
                best_policy[0],

            "phase_2":
                best_policy[1]
        })

        # ----------------------------------------------------
        # CREATE NEXT GENERATION
        # ----------------------------------------------------

        if generation < GENERATIONS:

            population = create_next_generation(
                population,
                fitnesses
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()
    print(
        "Saving evolutionary results..."
    )

    save_results(
        results
    )

    save_best_policy(
        global_best_policy,
        global_best_fitness,
        global_best_metrics["waiting"],
        global_best_metrics["queue"],
        global_best_metrics["vehicles"],
        global_best_metrics["throughput"]
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("EVOLUTIONARY OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(
        f"Generations      : "
        f"{GENERATIONS}"
    )

    print(
        f"Population       : "
        f"{POPULATION_SIZE}"
    )

    print(
        f"Best Policy      : "
        f"{global_best_policy}"
    )

    print(
        f"Best Fitness     : "
        f"{global_best_fitness:.3f}"
    )

    print(
        f"Best Waiting     : "
        f"{global_best_metrics['waiting']:.3f}"
    )

    print(
        f"Best Queue       : "
        f"{global_best_metrics['queue']:.3f}"
    )

    print(
        f"Best Vehicles    : "
        f"{global_best_metrics['vehicles']:.3f}"
    )

    print(
        f"Best Throughput  : "
        f"{global_best_metrics['throughput']}"
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
        "Best policy saved to:"
    )

    print(
        BEST_POLICY_FILE
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()