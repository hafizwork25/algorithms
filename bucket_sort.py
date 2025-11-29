def bucket_sort(arr):
    """
    Bucket Sort Algorithm
    Time Complexity: O(n + k) average case, O(n^2) worst case
    Space Complexity: O(n + k)
    """
    if len(arr) == 0:
        return arr

    bucket_count = len(arr)
    max_value = max(arr)
    min_value = min(arr)

    buckets = [[] for _ in range(bucket_count)]

    for num in arr:
        index = int((num - min_value) * (bucket_count - 1) / (max_value - min_value))
        buckets[index].append(num)

    for bucket in buckets:
        bucket.sort()

    sorted_array = []
    for bucket in buckets:
        sorted_array.extend(bucket)

    return sorted_array


if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    sorted_array = bucket_sort(test_array.copy())
    print("Sorted array:", sorted_array)
