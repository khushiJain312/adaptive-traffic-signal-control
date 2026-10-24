import os
import pandas as pd

from config import DATA_DIR, DATASET_FILE


class TrafficDataCollector:
    """
    Collects and stores traffic simulation data.
    """

    def __init__(self):

        self.records = []

    # ========================================================
    # ADD RECORD
    # ========================================================

    def add_record(
        self,
        episode,
        step,
        queue_length,
        waiting_time,
        vehicles,
        phase,
        action,
        throughput,
        reward,
        epsilon
    ):

        record = {
            "Episode": episode,
            "Step": step,
            "Queue_Length": queue_length,
            "Waiting_Time": waiting_time,
            "Vehicles": vehicles,
            "Traffic_Phase": phase,
            "Action": action,
            "Throughput": throughput,
            "Reward": reward,
            "Epsilon": epsilon
        }

        self.records.append(record)

    # ========================================================
    # GET DATAFRAME
    # ========================================================

    def get_dataframe(self):

        return pd.DataFrame(
            self.records
        )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    def save(self, filename=DATASET_FILE):

        if not self.records:

            print(
                "No traffic data available to save."
            )

            return None

        os.makedirs(
            DATA_DIR,
            exist_ok=True
        )

        df = self.get_dataframe()

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Traffic dataset saved to:\n{filename}"
        )

        return filename

    # ========================================================
    # CLEAR DATA
    # ========================================================

    def clear(self):

        self.records = []

    # ========================================================
    # NUMBER OF RECORDS
    # ========================================================

    def size(self):

        return len(
            self.records
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    collector = TrafficDataCollector()

    collector.add_record(
        episode=1,
        step=1,
        queue_length=5,
        waiting_time=10.5,
        vehicles=20,
        phase=0,
        action=1,
        throughput=3,
        reward=-7.5,
        epsilon=1.0
    )

    print(
        collector.get_dataframe()
    )

    print(
        f"Records: {collector.size()}"
    )