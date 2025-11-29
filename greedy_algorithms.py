import heapq
from collections import defaultdict


class Activity:
    """
    Represents an activity with start and finish times
    """

    def __init__(self, name, start, finish):
        self.name = name
        self.start = start
        self.finish = finish

    def __repr__(self):
        return f"Activity({self.name}, {self.start}-{self.finish})"


def activity_selection_greedy(activities):
    """
    Activity Selection Problem using Greedy Algorithm

    Problem: Given a set of activities with start and finish times,
    select the maximum number of non-overlapping activities.

    Greedy Choice: Always select the activity that finishes earliest
    among the remaining activities compatible with previously selected ones.

    Time Complexity: O(n log n) for sorting, O(n) for selection
    Space Complexity: O(n)

    Args:
        activities: List of Activity objects

    Returns:
        List of selected activities
    """
    if not activities:
        return []

    # Sort activities by finish time (greedy choice)
    sorted_activities = sorted(activities, key=lambda x: x.finish)

    selected = [sorted_activities[0]]
    last_finish_time = sorted_activities[0].finish

    for i in range(1, len(sorted_activities)):
        # Select activity if it starts after or when the last selected activity finishes
        if sorted_activities[i].start >= last_finish_time:
            selected.append(sorted_activities[i])
            last_finish_time = sorted_activities[i].finish

    return selected


def activity_selection_recursive(activities, k=0, n=None):
    """
    Activity Selection using Recursive Greedy Algorithm

    Time Complexity: O(n)
    Assumes activities are already sorted by finish time

    Args:
        activities: List of Activity objects (sorted by finish time)
        k: Index of last selected activity
        n: Total number of activities

    Returns:
        List of selected activities
    """
    if n is None:
        n = len(activities)

    if not activities:
        return []

    # Find the first activity that starts after activity k finishes
    m = k + 1
    while m < n and activities[m].start < activities[k].finish:
        m += 1

    if m < n:
        result = [activities[m]] + activity_selection_recursive(activities, m, n)
        return result
    else:
        return []


def fractional_knapsack(items, capacity):
    """
    Fractional Knapsack Problem using Greedy Algorithm

    Problem: Given items with weights and values, and a knapsack with capacity,
    maximize the total value. Items can be broken into fractions.

    Greedy Choice: Select items with highest value-to-weight ratio first

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        items: List of tuples (value, weight, name)
        capacity: Maximum weight capacity

    Returns:
        tuple: (max_value, selected_items with fractions)
    """
    # Calculate value-to-weight ratio for each item
    item_ratios = []
    for value, weight, name in items:
        ratio = value / weight if weight > 0 else 0
        item_ratios.append((ratio, value, weight, name))

    # Sort by ratio in descending order (greedy choice)
    item_ratios.sort(reverse=True)

    total_value = 0.0
    selected_items = []
    remaining_capacity = capacity

    for ratio, value, weight, name in item_ratios:
        if remaining_capacity == 0:
            break

        if weight <= remaining_capacity:
            # Take the whole item
            total_value += value
            selected_items.append((name, 1.0, value))
            remaining_capacity -= weight
        else:
            # Take fraction of the item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            selected_items.append((name, fraction, value * fraction))
            remaining_capacity = 0

    return total_value, selected_items


class HuffmanNode:
    """
    Node for Huffman Tree
    """

    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_coding(text):
    """
    Huffman Coding Algorithm for data compression

    Problem: Generate optimal prefix-free binary codes for characters
    based on their frequencies.

    Greedy Choice: Always combine two nodes with smallest frequencies

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        text: Input string

    Returns:
        dict: Character to binary code mapping
    """
    if not text:
        return {}

    # Count frequency of each character
    frequency = defaultdict(int)
    for char in text:
        frequency[char] += 1

    # Create a min heap of nodes
    heap = []
    for char, freq in frequency.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)

    # Build Huffman tree
    while len(heap) > 1:
        # Extract two nodes with minimum frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create internal node with frequency = sum of both nodes
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    # Generate codes by traversing the tree
    root = heap[0]
    codes = {}

    def generate_codes(node, current_code=""):
        if node is None:
            return

        if node.char is not None:
            codes[node.char] = current_code if current_code else "0"
            return

        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")

    generate_codes(root)
    return codes


def coin_change_greedy(coins, amount):
    """
    Coin Change Problem using Greedy Algorithm

    Problem: Make change for a given amount using minimum number of coins.

    Greedy Choice: Always use the largest coin that doesn't exceed remaining amount

    Note: This greedy approach works for canonical coin systems (like US coins),
    but may not give optimal solution for all coin systems.

    Time Complexity: O(n) where n is number of coin denominations
    Space Complexity: O(1)

    Args:
        coins: List of coin denominations (sorted in descending order)
        amount: Target amount

    Returns:
        tuple: (coin_count, list of coins used)
    """
    coins_sorted = sorted(coins, reverse=True)
    coin_count = 0
    coins_used = []
    remaining = amount

    for coin in coins_sorted:
        if remaining == 0:
            break

        if coin <= remaining:
            count = remaining // coin
            coin_count += count
            coins_used.extend([coin] * count)
            remaining -= coin * count

    if remaining > 0:
        return None, []  # Cannot make exact change

    return coin_count, coins_used


def interval_scheduling_maximization(intervals):
    """
    Interval Scheduling Maximization Problem

    Problem: Given a set of intervals, select maximum number of
    non-overlapping intervals.

    Similar to activity selection but works with generic intervals.

    Greedy Choice: Select interval with earliest finish time

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        intervals: List of tuples (start, end, name)

    Returns:
        List of selected intervals
    """
    if not intervals:
        return []

    # Sort by finish time
    sorted_intervals = sorted(intervals, key=lambda x: x[1])

    selected = [sorted_intervals[0]]
    last_end = sorted_intervals[0][1]

    for i in range(1, len(sorted_intervals)):
        start, end, name = sorted_intervals[i]
        if start >= last_end:
            selected.append(sorted_intervals[i])
            last_end = end

    return selected


def job_sequencing_with_deadlines(jobs):
    """
    Job Sequencing with Deadlines Problem

    Problem: Given jobs with deadlines and profits, schedule jobs to
    maximize profit. Each job takes 1 unit of time.

    Greedy Choice: Consider jobs in decreasing order of profit

    Time Complexity: O(n^2)
    Space Complexity: O(n)

    Args:
        jobs: List of tuples (job_id, deadline, profit)

    Returns:
        tuple: (total_profit, list of scheduled jobs)
    """
    if not jobs:
        return 0, []

    # Sort jobs by profit in descending order
    sorted_jobs = sorted(jobs, key=lambda x: x[2], reverse=True)

    # Find maximum deadline
    max_deadline = max(job[1] for job in jobs)

    # Create time slots
    slots = [None] * max_deadline
    total_profit = 0
    scheduled_jobs = []

    for job_id, deadline, profit in sorted_jobs:
        # Find a free slot for this job (starting from deadline)
        for t in range(min(deadline, max_deadline) - 1, -1, -1):
            if slots[t] is None:
                slots[t] = job_id
                total_profit += profit
                scheduled_jobs.append((job_id, t + 1, profit))
                break

    return total_profit, scheduled_jobs


def minimum_platforms(arrivals, departures):
    """
    Minimum Platforms Problem

    Problem: Given arrival and departure times of trains,
    find minimum number of platforms needed.

    Greedy Choice: Process events chronologically

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Args:
        arrivals: List of arrival times
        departures: List of departure times

    Returns:
        int: Minimum number of platforms needed
    """
    arrivals_sorted = sorted(arrivals)
    departures_sorted = sorted(departures)

    platforms_needed = 0
    max_platforms = 0
    i = 0
    j = 0
    n = len(arrivals)

    while i < n and j < n:
        if arrivals_sorted[i] <= departures_sorted[j]:
            platforms_needed += 1
            max_platforms = max(max_platforms, platforms_needed)
            i += 1
        else:
            platforms_needed -= 1
            j += 1

    return max_platforms


if __name__ == "__main__":
    print("=" * 80)
    print("GREEDY ALGORITHMS IMPLEMENTATION")
    print("=" * 80)

    # 1. Activity Selection Problem
    print("\n1. ACTIVITY SELECTION PROBLEM")
    print("-" * 80)
    print("Problem: Select maximum number of non-overlapping activities")
    print("Greedy Strategy: Always choose activity that finishes earliest\n")

    activities = [
        Activity("A1", 1, 4),
        Activity("A2", 3, 5),
        Activity("A3", 0, 6),
        Activity("A4", 5, 7),
        Activity("A5", 3, 9),
        Activity("A6", 5, 9),
        Activity("A7", 6, 10),
        Activity("A8", 8, 11),
        Activity("A9", 8, 12),
        Activity("A10", 2, 14),
        Activity("A11", 12, 16),
    ]

    print("Available activities:")
    for activity in activities:
        print(f"  {activity}")

    selected = activity_selection_greedy(activities)
    print(f"\nMaximum non-overlapping activities: {len(selected)}")
    print("Selected activities:")
    for activity in selected:
        print(f"  {activity}")

    # Recursive version
    sorted_activities = sorted(activities, key=lambda x: x.finish)
    selected_recursive = [sorted_activities[0]] + activity_selection_recursive(
        sorted_activities, 0, len(sorted_activities)
    )
    print(f"\nRecursive solution: {len(selected_recursive)} activities")

    # 2. Fractional Knapsack Problem
    print("\n\n2. FRACTIONAL KNAPSACK PROBLEM")
    print("-" * 80)
    print("Problem: Maximize value in knapsack (items can be fractioned)")
    print("Greedy Strategy: Choose items with highest value-to-weight ratio\n")

    items = [
        (60, 10, "Item1"),  # (value, weight, name)
        (100, 20, "Item2"),
        (120, 30, "Item3"),
        (80, 15, "Item4"),
    ]
    capacity = 50

    print(f"Knapsack capacity: {capacity}")
    print("Available items:")
    for value, weight, name in items:
        ratio = value / weight
        print(f"  {name}: value={value}, weight={weight}, ratio={ratio:.2f}")

    max_value, selected_items = fractional_knapsack(items, capacity)
    print(f"\nMaximum value: {max_value:.2f}")
    print("Selected items:")
    for name, fraction, value in selected_items:
        print(f"  {name}: {fraction * 100:.1f}% (value: {value:.2f})")

    # 3. Huffman Coding
    print("\n\n3. HUFFMAN CODING")
    print("-" * 80)
    print("Problem: Generate optimal prefix-free binary codes")
    print("Greedy Strategy: Combine two least frequent characters\n")

    text = "ABRACADABRA"
    print(f"Input text: {text}")

    # Count frequencies
    freq = defaultdict(int)
    for char in text:
        freq[char] += 1

    print("\nCharacter frequencies:")
    for char, count in sorted(freq.items()):
        print(f"  '{char}': {count}")

    codes = huffman_coding(text)
    print("\nHuffman codes:")
    for char, code in sorted(codes.items()):
        print(f"  '{char}': {code}")

    # Calculate compression ratio
    original_bits = len(text) * 8  # ASCII encoding
    compressed_bits = sum(len(codes[char]) for char in text)
    print(f"\nOriginal size: {original_bits} bits")
    print(f"Compressed size: {compressed_bits} bits")
    print(f"Compression ratio: {(1 - compressed_bits / original_bits) * 100:.1f}%")

    # 4. Coin Change Problem
    print("\n\n4. COIN CHANGE PROBLEM (GREEDY)")
    print("-" * 80)
    print("Problem: Make change with minimum number of coins")
    print("Greedy Strategy: Use largest coin first\n")

    coins = [1, 5, 10, 25, 50, 100]  # US coin system
    amount = 289

    print(f"Available coins: {coins}")
    print(f"Amount to make: {amount} cents")

    coin_count, coins_used = coin_change_greedy(coins, amount)
    print(f"\nMinimum coins needed: {coin_count}")
    print(f"Coins used: {coins_used}")

    # 5. Interval Scheduling
    print("\n\n5. INTERVAL SCHEDULING MAXIMIZATION")
    print("-" * 80)
    print("Problem: Select maximum non-overlapping intervals")
    print("Greedy Strategy: Choose interval with earliest finish time\n")

    intervals = [
        (1, 3, "I1"),
        (2, 4, "I2"),
        (3, 5, "I3"),
        (4, 6, "I4"),
        (5, 7, "I5"),
        (6, 8, "I6"),
    ]

    print("Available intervals:")
    for start, end, name in intervals:
        print(f"  {name}: [{start}, {end}]")

    selected_intervals = interval_scheduling_maximization(intervals)
    print(f"\nMaximum non-overlapping intervals: {len(selected_intervals)}")
    print("Selected intervals:")
    for start, end, name in selected_intervals:
        print(f"  {name}: [{start}, {end}]")

    # 6. Job Sequencing with Deadlines
    print("\n\n6. JOB SEQUENCING WITH DEADLINES")
    print("-" * 80)
    print("Problem: Schedule jobs to maximize profit")
    print("Greedy Strategy: Consider jobs in decreasing order of profit\n")

    jobs = [
        ("J1", 2, 100),  # (job_id, deadline, profit)
        ("J2", 1, 19),
        ("J3", 2, 27),
        ("J4", 1, 25),
        ("J5", 3, 15),
    ]

    print("Available jobs:")
    for job_id, deadline, profit in jobs:
        print(f"  {job_id}: deadline={deadline}, profit={profit}")

    total_profit, scheduled = job_sequencing_with_deadlines(jobs)
    print(f"\nMaximum profit: {total_profit}")
    print("Scheduled jobs:")
    for job_id, time_slot, profit in scheduled:
        print(f"  {job_id} at time slot {time_slot} (profit: {profit})")

    # 7. Minimum Platforms
    print("\n\n7. MINIMUM PLATFORMS PROBLEM")
    print("-" * 80)
    print("Problem: Find minimum platforms needed for trains")
    print("Greedy Strategy: Process arrival/departure events chronologically\n")

    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]

    print("Train schedule:")
    for i in range(len(arrivals)):
        print(f"  Train {i + 1}: arrives {arrivals[i]}, departs {departures[i]}")

    min_platforms = minimum_platforms(arrivals, departures)
    print(f"\nMinimum platforms needed: {min_platforms}")

    print("\n" + "=" * 80)
    print("All greedy algorithms completed successfully!")
    print("=" * 80)
    print("\nKey Insight: Greedy algorithms make locally optimal choices")
    print("at each step with the hope of finding a global optimum.")
    print("They work when the problem has optimal substructure and")
    print("the greedy choice property.")
    print("=" * 80)
