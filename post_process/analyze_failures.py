import json

fail_counts = {}
total_counts = {}

# Read and analyze the result file
with open('archieve/result.jsonl', 'r') as file:
    for line in file:
        jl = json.loads(line)
        attack = jl['attack']
        
        if attack not in fail_counts:
            fail_counts[attack] = 0
            total_counts[attack] = 0
        if jl['fail']:
            fail_counts[attack] += 1
        total_counts[attack] += 1

# Print failure rates
print("\nFailure rates by attack method:")
print("-" * 30)
for attack in fail_counts:
    rate = (fail_counts[attack] / total_counts[attack]) * 100
    print(f"{attack}: {rate:.2f}% ({fail_counts[attack]}/{total_counts[attack]})")
