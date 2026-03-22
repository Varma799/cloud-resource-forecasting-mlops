import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

rows = []
start_time = datetime(2026, 1, 1, 0, 0, 0)

servers = {
    "server_1": {"cpu_base": 45, "mem_base": 58, "req_base": 1400},
    "server_2": {"cpu_base": 55, "mem_base": 65, "req_base": 1700},
    "server_3": {"cpu_base": 30, "mem_base": 45, "req_base": 900},
}

for hour in range(40):
    current_time = start_time + timedelta(hours=hour)

    for server_id, profile in servers.items():
        cpu = profile["cpu_base"] + random.randint(-5, 8)
        memory = profile["mem_base"] + random.randint(-4, 6)
        disk_io = 100 + random.randint(0, 60)
        network_in = 200 + random.randint(20, 180)
        network_out = 180 + random.randint(20, 160)
        request_count = profile["req_base"] + random.randint(-150, 220)

        if server_id == "server_2" and hour % 9 == 0:
            cpu += 18
            memory += 12
            request_count += 300

        rows.append(
            {
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "server_id": server_id,
                "cpu_usage": round(min(max(cpu, 5), 98), 2),
                "memory_usage": round(min(max(memory, 10), 98), 2),
                "disk_io": round(disk_io, 2),
                "network_in": round(network_in, 2),
                "network_out": round(network_out, 2),
                "request_count": int(max(request_count, 100)),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("data/raw/cloud_resource_usage.csv", index=False)

print(df.shape)
print(df.head())