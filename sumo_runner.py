import os
import sys
import traci

from config import (
    SUMO_BINARY,
    SUMO_GUI_BINARY,
    SUMO_CONFIG,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
    YELLOW_TIME
)


# ============================================================
# ADD SUMO TO PYTHON PATH
# ============================================================

if "SUMO_HOME" in os.environ:

    tools = os.path.join(
        os.environ["SUMO_HOME"],
        "tools"
    )

    if tools not in sys.path:

        sys.path.append(tools)


# ============================================================
# SUMO RUNNER
# ============================================================

class SUMORunner:

    def __init__(
        self,
        use_gui=False,
        step_length=1
    ):

        self.use_gui = use_gui
        self.step_length = step_length

        self.sumo_binary = (
            SUMO_GUI_BINARY
            if use_gui
            else SUMO_BINARY
        )

        self.sumo_config = SUMO_CONFIG

        self.current_step = 0

        self.traffic_light_id = None

        self.last_phase = None
        self.last_phase_change = 0

        self.connected = False

        # Vehicles arrived in the previous step
        self.previous_arrived = 0


    # ========================================================
    # START SUMO
    # ========================================================

    def start(self):

        if not os.path.exists(
            self.sumo_binary
        ):

            raise FileNotFoundError(
                f"SUMO executable not found:\n"
                f"{self.sumo_binary}"
            )

        if not os.path.exists(
            self.sumo_config
        ):

            raise FileNotFoundError(
                f"SUMO configuration not found:\n"
                f"{self.sumo_config}"
            )

        command = [
            self.sumo_binary,
            "-c",
            self.sumo_config,
            "--step-length",
            str(self.step_length),
            "--start"
        ]

        print("=" * 60)
        print("Starting SUMO")
        print("=" * 60)

        print(
            "SUMO:",
            self.sumo_binary
        )

        print(
            "Config:",
            self.sumo_config
        )

        print(
            "GUI:",
            self.use_gui
        )

        print(
            "Connecting through TraCI..."
        )

        try:

            traci.start(command)

            self.connected = True
            self.current_step = 0
            self.previous_arrived = (
                traci.simulation.getArrivedNumber()
            )

            traffic_lights = (
                traci.trafficlight.getIDList()
            )

            if traffic_lights:

                self.traffic_light_id = (
                    traffic_lights[0]
                )

                self.last_phase = (
                    traci.trafficlight.getPhase(
                        self.traffic_light_id
                    )
                )

                self.last_phase_change = 0

                print(
                    "Traffic light:",
                    self.traffic_light_id
                )

                print(
                    "Initial phase:",
                    self.last_phase
                )

            else:

                print(
                    "Warning: No traffic light found."
                )

        except Exception as e:

            self.connected = False

            print(
                "Failed to start SUMO:"
            )

            print(e)

            raise


    # ========================================================
    # CLOSE SUMO
    # ========================================================

    def close(self):

        if self.connected:

            try:

                traci.close()

            except Exception:
                pass

        self.connected = False
        self.traffic_light_id = None
        self.current_step = 0


    # ========================================================
    # STEP SIMULATION
    # ========================================================

    def step(self):

        if not self.connected:
            return False

        try:

            traci.simulationStep()

            self.current_step += 1

            return True

        except Exception as e:

            print(
                "Simulation step error:",
                e
            )

            return False


    # ========================================================
    # CURRENT STEP
    # ========================================================

    def get_current_step(self):

        return self.current_step


    # ========================================================
    # TRAFFIC LIGHT PHASE
    # ========================================================

    def get_phase(self):

        if not self.connected:
            return 0

        if self.traffic_light_id is None:
            return 0

        try:

            return traci.trafficlight.getPhase(
                self.traffic_light_id
            )

        except Exception:

            return 0


    # ========================================================
    # SET TRAFFIC LIGHT PHASE
    # ========================================================

    def set_phase(self, phase):

        if not self.connected:
            return

        if self.traffic_light_id is None:
            return

        try:

            traci.trafficlight.setPhase(
                self.traffic_light_id,
                phase
            )

            self.last_phase = phase

            self.last_phase_change = (
                self.current_step
            )

        except Exception as e:

            print(
                "Phase change error:",
                e
            )


    # ========================================================
    # VEHICLE COUNT
    # ========================================================

    def get_vehicle_count(self):

        if not self.connected:
            return 0

        try:

            return len(
                traci.vehicle.getIDList()
            )

        except Exception:

            return 0


    # ========================================================
    # QUEUE LENGTH
    # ========================================================

    def get_queue_length(self):

        if not self.connected:
            return 0

        try:

            vehicles = (
                traci.vehicle.getIDList()
            )

        except Exception:

            return 0

        queue = 0

        for vehicle_id in vehicles:

            try:

                speed = (
                    traci.vehicle.getSpeed(
                        vehicle_id
                    )
                )

                if speed < 0.1:

                    queue += 1

            except Exception:
                pass

        return queue


    # ========================================================
    # WAITING TIME
    # ========================================================

    def get_waiting_time(self):

        if not self.connected:
            return 0.0

        try:

            vehicles = (
                traci.vehicle.getIDList()
            )

        except Exception:

            return 0.0

        total_waiting = 0.0

        for vehicle_id in vehicles:

            try:

                total_waiting += (
                    traci.vehicle.getWaitingTime(
                        vehicle_id
                    )
                )

            except Exception:
                pass

        return total_waiting


    # ========================================================
    # AVERAGE SPEED
    # ========================================================

    def get_average_speed(self):

        if not self.connected:
            return 0.0

        try:

            vehicles = (
                traci.vehicle.getIDList()
            )

        except Exception:

            return 0.0

        if not vehicles:
            return 0.0

        total_speed = 0.0
        count = 0

        for vehicle_id in vehicles:

            try:

                total_speed += (
                    traci.vehicle.getSpeed(
                        vehicle_id
                    )
                )

                count += 1

            except Exception:
                pass

        if count == 0:
            return 0.0

        return total_speed / count


    # ========================================================
    # VEHICLE WAITING TIME
    # ========================================================

    def get_vehicle_waiting_time(
        self,
        vehicle_id
    ):

        if not self.connected:
            return 0.0

        try:

            return traci.vehicle.getWaitingTime(
                vehicle_id
            )

        except Exception:

            return 0.0


    # ========================================================
    # VEHICLE SPEED
    # ========================================================

    def get_vehicle_speed(
        self,
        vehicle_id
    ):

        if not self.connected:
            return 0.0

        try:

            return traci.vehicle.getSpeed(
                vehicle_id
            )

        except Exception:

            return 0.0


    # ========================================================
    # VEHICLE POSITION
    # ========================================================

    def get_vehicle_position(
        self,
        vehicle_id
    ):

        if not self.connected:
            return None

        try:

            return traci.vehicle.getPosition(
                vehicle_id
            )

        except Exception:

            return None


    # ========================================================
    # VEHICLE LANE
    # ========================================================

    def get_vehicle_lane(
        self,
        vehicle_id
    ):

        if not self.connected:
            return ""

        try:

            return traci.vehicle.getLaneID(
                vehicle_id
            )

        except Exception:

            return ""


    # ========================================================
    # CUMULATIVE THROUGHPUT
    # ========================================================

    def get_throughput(self):

        if not self.connected:
            return 0

        try:

            return traci.simulation.getArrivedNumber()

        except Exception:

            return 0


    # ========================================================
    # STEP THROUGHPUT
    # ========================================================

    def get_step_throughput(self):

        """
        Number of vehicles that arrived since the
        previous traffic-state measurement.
        """

        if not self.connected:
            return 0

        try:

            current_arrived = (
                traci.simulation.getArrivedNumber()
            )

            step_arrived = (
                current_arrived
                - self.previous_arrived
            )

            self.previous_arrived = (
                current_arrived
            )

            return max(
                0,
                step_arrived
            )

        except Exception:

            return 0


    # ========================================================
    # SIMULATION TIME
    # ========================================================

    def get_simulation_time(self):

        if not self.connected:
            return 0.0

        try:

            return traci.simulation.getTime()

        except Exception:

            return 0.0


    # ========================================================
    # SIGNAL PARAMETERS
    # ========================================================

    def get_min_green_time(self):

        return MIN_GREEN_TIME


    def get_max_green_time(self):

        return MAX_GREEN_TIME


    def get_yellow_time(self):

        return YELLOW_TIME


    # ========================================================
    # TRAFFIC LIGHT ID
    # ========================================================

    def get_traffic_light_id(self):

        return self.traffic_light_id


    # ========================================================
    # CONNECTION STATUS
    # ========================================================

    def is_connected(self):

        return self.connected


    # ========================================================
    # FULL TRAFFIC STATE
    # ========================================================

    def get_traffic_state(self):

        if not self.connected:

            return {
                "vehicles": 0,
                "queue": 0,
                "waiting_time": 0.0,
                "average_speed": 0.0,
                "phase": 0,
                "throughput": 0
            }

        return {

            "vehicles":
                self.get_vehicle_count(),

            "queue":
                self.get_queue_length(),

            "waiting_time":
                self.get_waiting_time(),

            "average_speed":
                self.get_average_speed(),

            "phase":
                self.get_phase(),

            "throughput":
                self.get_throughput()
        }


    # ========================================================
    # RL STATE
    # ========================================================

    def get_state(self):

        """
        Return the raw traffic measurements needed
        by the QLearningAgent.

        The agent performs the actual discretization.
        """

        if not self.connected:

            return (
                0,
                0,
                0
            )

        try:

            return (
                self.get_queue_length(),
                self.get_waiting_time(),
                self.get_phase()
            )

        except Exception as e:

            print(
                "State calculation error:",
                e
            )

            return (
                0,
                0,
                0
            )


    # ========================================================
    # PRINT STATUS
    # ========================================================

    def print_status(self):

        if not self.connected:

            print(
                "SUMO is not connected."
            )

            return

        print(
            f"Step: {self.current_step} | "
            f"Vehicles: {self.get_vehicle_count()} | "
            f"Queue: {self.get_queue_length()} | "
            f"Waiting: {self.get_waiting_time():.2f} | "
            f"Throughput: {self.get_throughput()}"
        )