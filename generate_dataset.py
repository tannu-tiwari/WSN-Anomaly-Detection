import subprocess
import itertools

NS3_DIR = "/home/tannu/WSN-Project/ns-3-dev"

START_RUN = 18
END_RUN = 25

distances = [40, 42, 44, 45, 46, 48, 50, 55, 60]
packet_sizes = [256, 512, 1024]
intervals = [0.25, 0.5, 1.0]

run_id = START_RUN

for distance, packet_size, interval in itertools.product(
    distances,
    packet_sizes,
    intervals
):
    if run_id > END_RUN:
        break

    print("=" * 60)
    print(f"Running simulation {run_id}")
    print(f"Distance    : {distance} m")
    print(f"Packet Size : {packet_size} bytes")
    print(f"Interval    : {interval} seconds")
    print("=" * 60)

    command = [
        "./ns3",
        "run",
        f"scratch/wsn-basic "
        f"--runId={run_id} "
        f"--scenario=normal "
        f"--nodeDistance={distance} "
        f"--packetSize={packet_size} "
        f"--interval={interval}"
    ]

    try:
        subprocess.run(command, cwd=NS3_DIR, check=True)
        print(f"Run {run_id} completed successfully.")

    except subprocess.CalledProcessError:
        print(f"Run {run_id} failed.")

    run_id += 1

print("=" * 60)
print("Dataset generation completed.")
print(f"Runs generated: {START_RUN} to {run_id - 1}")
print("=" * 60)
