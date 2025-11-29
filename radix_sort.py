def counting_sort(arr, exp):
    """
    Helper function for radix sort using counting sort
    """
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr):
    """
    Radix Sort Algorithm
    Time Complexity: O(d * (n + k)) where d is number of digits
    Space Complexity: O(n + k)
    """
    if len(arr) == 0:
        return arr

    max_value = max(arr)

    exp = 1
    while max_value // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

    return arr


if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    sorted_array = radix_sort(test_array.copy())
    print("Sorted array:", sorted_array)
